import numpy as np
import logging
import torch.backends.cudnn as cudnn
import os
import pickle
from method.model import MS_SL_Net
from torch.utils.data import DataLoader
from method.data_provider import Dataset4MS_SL,VisDataSet4MS_SL,\
    TxtDataSet4MS_SL,read_video_ids, collate_frame_val, collate_text_val
from tqdm import tqdm
from collections import defaultdict
import torch
from utils.basic_utils import AverageMeter, BigFile, read_dict
from method.config import TestOptions
import csv

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def ap_score(sorted_labels):
    nr_relevant = len([x for x in sorted_labels if x > 0])
    if nr_relevant == 0:
        return 0.0

    length = len(sorted_labels)
    ap = 0.0
    rel = 0

    for i in range(length):
        lab = sorted_labels[i]
        if lab >= 1:
            rel += 1
            ap += float(rel) / (i + 1.0)
    ap /= nr_relevant
    return ap

def get_gt(video_metas, query_metas):
    v2t_gt = []
    for vid_id in video_metas:
        v2t_gt.append([])
        for i, query_id in enumerate(query_metas):
            if query_id.split('#', 1)[0] == vid_id:
                v2t_gt[-1].append(i)

    t2v_gt = {}
    for i, t_gts in enumerate(v2t_gt):
        for t_gt in t_gts:
            t2v_gt.setdefault(t_gt, [])
            t2v_gt[t_gt].append(i)

    return v2t_gt, t2v_gt

def eval_q2m(scores, q2m_gts):
    n_q, n_m = scores.shape

    gt_ranks = np.zeros((n_q,), np.int32)
    aps = np.zeros(n_q)
    for i in range(n_q):
        s = scores[i]
        sorted_idxs = np.argsort(s)
        rank = n_m + 1
        tmp_set = []
        for k in q2m_gts[i]:
            tmp = np.where(sorted_idxs == k)[0][0] + 1
            if tmp < rank:
                rank = tmp

        gt_ranks[i] = rank

    # compute metrics
    r1 = 100.0 * len(np.where(gt_ranks <= 1)[0]) / n_q
    r5 = 100.0 * len(np.where(gt_ranks <= 5)[0]) / n_q
    r10 = 100.0 * len(np.where(gt_ranks <= 10)[0]) / n_q
    r100 = 100.0 * len(np.where(gt_ranks <= 100)[0]) / n_q
    medr = np.median(gt_ranks)
    meanr = gt_ranks.mean()

    return (r1, r5, r10, r100, medr, meanr), gt_ranks

def t2v_map(c2i, t2v_gts):
    perf_list = []
    for i in range(c2i.shape[0]):
        d_i = c2i[i, :]
        labels = [0]*len(d_i)

        x = t2v_gts[i][0]
        labels[x] = 1

        sorted_labels = [labels[x] for x in np.argsort(d_i)]

        current_score = ap_score(sorted_labels)
        perf_list.append(current_score)
    return np.mean(perf_list)


def compute_context_info(model, eval_dataset, opt): # 进来的是video_feature(global caption放在最后了)
    model.eval()
    n_total_vid = len(eval_dataset)
    context_dataloader = DataLoader(eval_dataset, collate_fn=collate_frame_val, batch_size=opt.eval_context_bsz,
                                    num_workers=opt.num_workers, shuffle=False, pin_memory=opt.pin_memory)
    # import ipdb;ipdb.set_trace()
    # clip_videos, frame_videos, videos_mask, idxs, video_ids,global_captions,global_caption_mask
    bsz = opt.eval_context_bsz
    metas = []  # list(dicts) video_ids
    frame_feat, frame_mask = [], []
    global_caption_feat= []  # 用于存储每个 batch 中的 _global_captions

    vid_proposal_feat = None
    
    for idx, batch in tqdm(enumerate(context_dataloader), desc="Computing query2video scores",
                           total=len(context_dataloader)):
        clip_videos, frame_videos, videos_mask, idxs, video_ids=batch     # ,global_captions,global_caption_mask
        metas.extend(video_ids)
        
        # global_captions_ = global_captions.to(opt.device) # global_captions[200, padding_num,1024]
        # global_caption_mask_ = global_caption_mask.to(opt.device) # global_captions[200, padding_num]
        # _global_captions, _ = model.encode_query(global_captions_, global_caption_mask_) # encoder降维+avg pooling: [200,384]
        # global_caption_feat.append(_global_captions)

        clip_video_feat_ = clip_videos.to(opt.device) # clip_videos [200, 32, 1024]
        frame_video_feat_ = frame_videos.to(opt.device) # frame_videos [200, 128, 1024]
        frame_mask_ = videos_mask.to(opt.device) # videos_mask [video_ids,fea_len(128)] [200, 128]
        _frame_feat, _video_proposal_feat, _ = model.encode_context(clip_video_feat_, frame_video_feat_, frame_mask_) # encoder 降维（clip，frame）和滑动窗口（clip）
        _video_proposal_feat = _video_proposal_feat.cpu().numpy()

        # 每个batch组合起来
        frame_feat.append(_frame_feat) 
        frame_mask.append(frame_mask_)
        if vid_proposal_feat is None:
            vid_proposal_feat = np.zeros((n_total_vid, _video_proposal_feat.shape[1], opt.hidden_size),
                                         dtype=np.float32) #[video_ids,528,384]
            vid_proposal_feat[idx * bsz:(idx + 1) * bsz] = _video_proposal_feat
        else:
            vid_proposal_feat[idx * bsz:(idx + 1) * bsz] = _video_proposal_feat
    vid_proposal_feat = torch.from_numpy(vid_proposal_feat).to(opt.device)
    def cat_tensor(tensor_list):
        if len(tensor_list) == 0:
            return None
        else:
            seq_l = [e.shape[1] for e in tensor_list]
            b_sizes = [e.shape[0] for e in tensor_list]
            b_sizes_cumsum = np.cumsum([0] + b_sizes)
            if len(tensor_list[0].shape) == 3:
                hsz = tensor_list[0].shape[2]
                res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l), hsz)
            elif len(tensor_list[0].shape) == 2:
                res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l))
            else:
                raise ValueError("Only support 2/3 dimensional tensors")
            for i, e in enumerate(tensor_list):
                res_tensor[b_sizes_cumsum[i]:b_sizes_cumsum[i+1], :seq_l[i]] = e
            return res_tensor

    return dict(
        video_metas=metas,  # list(dict) (N_videos)
        video_proposal_feat=vid_proposal_feat,
        video_feat=cat_tensor(frame_feat),
        video_mask=cat_tensor(frame_mask)
        ) #        global_caption_feat=cat_tensor(global_caption_feat)

def compute_query2ctx_info(model, eval_dataset, opt, ctx_info): # model, 测试集query(未拼接), opt, 测试集feature+global caption(已拼接)
    model.eval()
    query_eval_loader = DataLoader(eval_dataset, collate_fn=collate_text_val, batch_size=opt.eval_query_bsz,
                                   num_workers=opt.num_workers, shuffle=False, pin_memory=opt.pin_memory)
    # target, words_mask, idxs, cap_ids, raw_cap_ids

    query_metas = []
    query_metas_raw = []
    clip_scale_scores = []
    frame_scale_scores = []
    score_sum = []
    score_sum_only_query_cap = []
    best_clip_indices = []
    for idx, batch in tqdm(enumerate(query_eval_loader), desc="Computing q embedding", total=len(query_eval_loader)):
        target, words_mask, idxs, cap_ids, raw_cap_ids = batch
        query_metas.extend(cap_ids) # ['F7TG5', '9JZO2', '9JZO2', '4J1AP',...]
        query_metas_raw.extend(raw_cap_ids) # ['F7TG5#enc#3', '9JZO2#enc#3', '9JZO2#enc#10', '4J1AP#enc#1',...]
        query_feat = target.to(opt.device) # [bsz,max_token_num,1024]
        query_mask = words_mask.to(opt.device) # [bsz,max_token_num]
        # query_global_cap_scores = model.get_query_global_cap_scores(query_feat, query_mask,ctx_info["global_caption_feat"])
        _clip_scale_scores, _frame_scale_scores, key_clip_indices = model.get_pred_from_raw_query(
            query_feat, query_mask, None, ctx_info["video_proposal_feat"], ctx_info["video_feat"], ctx_info['video_mask'],
            )#cross=True
        _score_sum = opt.clip_scale_w*_clip_scale_scores + opt.frame_scale_w*_frame_scale_scores # + query_cap_weight * query_global_cap_scores # 两个weight都是0.5
        # _score_sum_only_query_cap = query_cap_weight * query_global_cap_scores
        # clip_scale_scores.append(_clip_scale_scores)
        # frame_scale_scores.append(_frame_scale_scores)
        # score_sum.append(_score_sum)
        #合并batch
        clip_scale_scores.append(_clip_scale_scores.cpu().numpy())
        frame_scale_scores.append(_frame_scale_scores.cpu().numpy())
        score_sum.append(_score_sum.cpu().numpy())
        # score_sum_only_query_cap.append(_score_sum_only_query_cap.cpu().numpy())
        best_clip_indices.append(key_clip_indices.cpu().numpy()) # [query_num_in_a_batch,video_nums_all]
        
        # 清除缓存
        del _clip_scale_scores, _frame_scale_scores, _score_sum #,_score_sum_only_query_cap
        torch.cuda.empty_cache()  # 可选，根据实际情况判断是否需要

    # clip_scale_scores = torch.cat(clip_scale_scores, dim=0).cpu().numpy().copy()
    # frame_scale_scores = torch.cat(frame_scale_scores, dim=0).cpu().numpy().copy()
    # score_sum = torch.cat(score_sum, dim=0).cpu().numpy().copy()
    # return clip_scale_scores, frame_scale_scores, score_sum, query_metas
    return (
        np.concatenate(clip_scale_scores, axis=0),  # 直接拼接 NumPy 数组
        np.concatenate(frame_scale_scores, axis=0),
        np.concatenate(score_sum, axis=0),
        np.concatenate(best_clip_indices, axis=0),   #   np.concatenate(score_sum_only_query_cap, axis=0),
        query_metas,
        query_metas_raw
    )



def cal_perf(t2v_all_errors, t2v_gt):

    # video retrieval
    (t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr), gt_ranks = eval_q2m(t2v_all_errors, t2v_gt)
    # if score_sum_only_query_cap is not None:
    #     (q2c_r1, q2c_r5, q2c_r10, q2c_r100, _, _), _ = eval_q2m(score_sum_only_query_cap, t2v_gt)
    t2v_map_score = t2v_map(t2v_all_errors, t2v_gt)


    logging.info(" * Text to Video:")
    logging.info(" * r_1_5_10_100, medr, meanr: {}".format([round(t2v_r1, 1), round(t2v_r5, 1), round(t2v_r10, 1), round(t2v_r100, 1)]))
    logging.info(" * recall sum: {}".format(round(t2v_r1+t2v_r5+t2v_r10+t2v_r100, 1)))
    # if score_sum_only_query_cap is not None:
    #     logging.info(" * score_sum_only_query_cap: {}".format(round(q2c_r1+q2c_r5+q2c_r10+q2c_r100, 1)))
    logging.info(" * mAP: {}".format(round(t2v_map_score, 4)))
    logging.info(" * "+'-'*10)

    return (t2v_r1, t2v_r5, t2v_r10,t2v_r100, t2v_medr, t2v_meanr, t2v_map_score), gt_ranks


def eval_epoch(model, val_video_dataset, val_text_dataset, opt):
    model.eval()
    logger.info("Computing scores")
    context_info = compute_context_info(model, val_video_dataset, opt) # 提取测试集的video feature (clip, frame) 返回字典

    
    # 额外排名变化分析，不影响原有流程
    currscore = 0
    all_queryid2text = val_text_dataset.captions
    baseline_ranks = None
    baseline_query_metas_raw = None
    baseline_query_metas = None
    
    
    # for i in [5]:
    # query_cap_weight = i * 0.1
    # logger.info(f"query_cap_weight: {query_cap_weight}")
    query_context_scores, global_query_context_scores, score_sum, best_clip_indices,query_metas, query_metas_raw = compute_query2ctx_info(model, val_text_dataset, opt, context_info)
    video_metas = context_info['video_metas']
    v2t_gt, t2v_gt = get_gt(video_metas, query_metas)
    print('clip_scale_scores:')
    (clip_t2v_r1, clip_t2v_r5, clip_t2v_r10, clip_t2v_r100, clip_t2v_medr, clip_t2v_meanr, clip_t2v_map_score), clip_gt_ranks = cal_perf(-1 * query_context_scores, t2v_gt)

    print('frame_scale_scores:')
    (global_t2v_r1, global_t2v_r5, global_t2v_r10, global_t2v_r100, global_t2v_medr, global_t2v_meanr, global_t2v_map_score), global_gt_ranks = cal_perf(-1 * global_query_context_scores, t2v_gt)
    print('score_sum:')
    (t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr, t2v_map_score), gt_ranks = cal_perf(-1 * score_sum, t2v_gt)
    currscore = 0
    currscore += (t2v_r1 + t2v_r5 + t2v_r10 + t2v_r100)

    return currscore


def setup_model(opt):
    """Load model from checkpoint and move to specified device"""
    ckpt_filepath = os.path.join(opt.ckpt_filepath)
    checkpoint = torch.load(ckpt_filepath)
    loaded_model_cfg = checkpoint["model_cfg"]
    NAME_TO_MODELS = {'MS_SL_Net':MS_SL_Net}
    model = NAME_TO_MODELS[opt.model_name](loaded_model_cfg)
    
    model.load_state_dict(checkpoint["model"])
    logger.info("Loaded model saved at epoch {} from checkpoint: {}".format(checkpoint["epoch"], opt.ckpt_filepath))

    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        if len(opt.device_ids) > 1:
            logger.info("Use multi GPU", opt.device_ids)
            model = torch.nn.DataParallel(model, device_ids=opt.device_ids)  # use multi GPU
    return model

def start_inference(opt=None):
    logger.info("Setup config, data and model...")
    if opt is None:
        opt = TestOptions().parse()
    cudnn.benchmark = False
    cudnn.deterministic = True

    rootpath = opt.root_path
    collection = opt.collection
    testCollection = '%stest' % collection

    cap_file = {'test': '%s.caption.txt' % testCollection}

    # caption
    caption_files = {x: os.path.join(rootpath, collection, 'TextData', cap_file[x])
                     for x in cap_file}

    text_feat_path = os.path.join(rootpath, collection, 'TextData', 'roberta_%s_query_feat.hdf5' % collection)
    # Load visual features
    visual_feat_path = os.path.join(rootpath, collection, 'FeatureData', opt.visual_feature)

    visual_feats = BigFile(visual_feat_path)
    video2frames =  read_dict(os.path.join(rootpath, collection, 'FeatureData', opt.visual_feature, 'video2frames.txt'))


    test_video_ids_list = read_video_ids(caption_files['test'])
    test_vid_dataset = VisDataSet4MS_SL(visual_feats, video2frames, opt,
                                               video_ids=test_video_ids_list)
    test_text_dataset = TxtDataSet4MS_SL(caption_files['test'], text_feat_path, opt)
    # global_caption



    model = setup_model(opt)

    logger.info("Starting inference...")
    with torch.no_grad():
        score = eval_epoch(model, test_vid_dataset, test_text_dataset, opt)



def compute_query2ctx_info_cal_flops(model, eval_dataset, opt, ctx_info,frame_key,frame_value,top_k):
    """
    只处理eval_dataset中的第一个query，返回与compute_query2ctx_info相同格式的结果。
    用于FLOPs计算。
    """
    model.eval()

    query_eval_loader = DataLoader(eval_dataset, collate_fn=collate_text_val, batch_size=opt.eval_query_bsz,
                                num_workers=opt.num_workers, shuffle=False, pin_memory=opt.pin_memory)


    for batch in query_eval_loader:
        target, words_mask, idxs, cap_ids, raw_cap_ids = batch
        
        
        query_feat = target.to(opt.device)
        query_mask = words_mask.to(opt.device)
        query_feat_x=query_feat[:1]
        query_mask_x=query_mask[:1]
        if len(query_feat_x.shape) < 3: 
            query_feat_x = query_feat_x.unsqueeze(0)
            query_mask_x = query_mask_x.unsqueeze(0)
        
        _clip_scale_scores, _frame_scale_scores, key_clip_indices = model.get_pred_from_raw_query_cal_flops(
            query_feat_x, query_mask_x, None, ctx_info["video_proposal_feat"], ctx_info["video_feat"], ctx_info['video_mask'],False,None,frame_key,frame_value,top_k
        )
        
        

        # _score_sum = opt.clip_scale_w*_clip_scale_scores + opt.frame_scale_w*_frame_scale_scores
        
        break



def eval_for_cal_flops(model, val_video_dataset, val_text_dataset, opt,context_info,frame_key,frame_value,top_k):
    model.eval()
    logger.info("Computing scores")

    
    # 额外排名变化分析，不影响原有流程
    currscore = 0
    
    
    # for i in [5]:
    # query_cap_weight = i * 0.1
    # logger.info(f"query_cap_weight: {query_cap_weight}")
    compute_query2ctx_info_cal_flops(model, val_text_dataset, opt, context_info,frame_key,frame_value,top_k)



if __name__ == '__main__':
    start_inference()