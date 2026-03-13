import numpy as np
import torch.nn.functional as F
import logging
import torch.backends.cudnn as cudnn
import os
import json
from sklearn.cluster import KMeans
import pickle
from method.model import MS_SL_Net
from torch.utils.data import DataLoader
from method.data_provider import Dataset4MS_SL,VisDataSet4MS_SL,\
    TxtDataSet4MS_SL,read_video_ids, collate_frame_val, collate_text_val,TxtDataSet4MS_SL_train
from tqdm import tqdm
from collections import defaultdict
import torch
from utils.basic_utils import AverageMeter, BigFile, read_dict
from method.config import TestOptions
import ipdb
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
from functools import partial

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
        s = scores[i] # 按行(query)取出score
        sorted_idxs = np.argsort(s) # 从小到大的排序索引
        rank = n_m + 1
        tmp_set = []
        for k in q2m_gts[i]:
            tmp = np.where(sorted_idxs == k)[0][0] + 1 # 找gt在排序中的index. 在模型输出中gt是第几小的
            # ipdb.set_trace()
            if tmp < rank:
                rank = tmp

        gt_ranks[i] = rank # gt_ranks[i] = 第 i 个 query 的 GT rank

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

def add_value_to_top_indices(TT_score, top_k, value=1000): # 对于baseline + TT 不是直接吧TT加进去,而是选前topk个train query的部分加进去 [test query, train query]
    dim = TT_score.shape[1]
    top_values, topk_indices = torch.topk(TT_score, k=int(dim*top_k), dim=1)
    processed_score = torch.full_like(TT_score, float("-inf"))
    dim1_indices = torch.arange(TT_score.shape[0], device=TT_score.device).unsqueeze(1)
    processed_score[dim1_indices, topk_indices] = TT_score[dim1_indices, topk_indices]
    
    return processed_score,topk_indices

def compute_context_info(model, eval_dataset, opt, need_caption = True): # 进来的是video_feature(global caption放在最后了)
    model.eval()
    n_total_vid = len(eval_dataset)
    if need_caption:
        context_dataloader = DataLoader(eval_dataset, collate_fn=collate_frame_val, batch_size=opt.eval_context_bsz,
                                        num_workers=opt.num_workers, shuffle=False, pin_memory=opt.pin_memory)
        global_caption_feat= []  # 用于存储每个 batch 中的 _global_captions
        all_start_end_list = []
        desc = "拼接测试集 video feature 和 caption"
    # clip_videos, frame_videos, videos_mask, idxs, labels, video_ids, global_captions,global_caption_mask                                        
    else:
        collate_fn_with_flag = partial(collate_frame_val, need_caption=False)
        context_dataloader = DataLoader(eval_dataset,collate_fn=collate_fn_with_flag,batch_size=opt.eval_context_bsz,
                                        num_workers=opt.num_workers, shuffle=False, pin_memory=opt.pin_memory)
        desc = "拼接训练集 video feature"
        
    # clip_videos, frame_videos, videos_mask, idxs, labels, video_ids, (global_captions,global_caption_mask)

    bsz = opt.eval_context_bsz
    metas = []  # list(dicts) video_ids
    labels_all = []
    frame_feat, frame_mask = [], []
    vid_proposal_feat = None
    
    for idx, batch in tqdm(enumerate(context_dataloader), desc=desc,total=len(context_dataloader)):
        if need_caption:
            clip_videos, frame_videos, videos_mask, idxs, labels, video_ids,global_captions,global_caption_mask,start_end_flattened=batch
            global_captions_ = global_captions.to(opt.device) # global_captions[200, padding_num,1024]
            global_caption_mask_ = global_caption_mask.to(opt.device) # global_captions[200, padding_num]
            _global_captions, _ = model.encode_query(global_captions_, global_caption_mask_) # encoder降维+avg pooling: [200,384]
            global_caption_feat.append(_global_captions)
            all_start_end_list.extend(start_end_flattened)
        else:
            clip_videos, frame_videos, videos_mask, idxs, labels, video_ids,_,_ = batch
                      
        metas.extend(video_ids)
        labels_all.extend(labels)# labels = [0, 0, 1, 1, 2, 2, 3...]

        clip_video_feat_ = clip_videos.to(opt.device) # clip_videos [200, 32, 1024]
        frame_video_feat_ = frame_videos.to(opt.device) # frame_videos [200, 128, 1024]
        frame_mask_ = videos_mask.to(opt.device) # videos_mask [video_ids,fea_len(128)] [200, 128]
        # ipdb.set_trace()
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
    start_end_tensor = torch.tensor(all_start_end_list).to(opt.device)
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
    if need_caption:
        return dict(
            video_metas=metas,  # list(dict) (N_videos)
            video_proposal_feat=vid_proposal_feat,
            video_feat=cat_tensor(frame_feat),
            video_mask=cat_tensor(frame_mask),
            global_caption_feat=cat_tensor(global_caption_feat),
            labels=labels_all,
            start_end_tensor=start_end_tensor
            )
    else:
        return dict(
            video_metas=metas,  # list(dict) (N_videos)
            video_proposal_feat=vid_proposal_feat,
            video_feat=cat_tensor(frame_feat),
            video_mask=cat_tensor(frame_mask),
            labels=labels_all,
            start_end_tensor=start_end_tensor
            )

def compute_query2ctx_info(model, eval_dataset, opt, ctx_info,TT_weight=0,top_k=0,local_weight=0,train_text_dataset=None,context_info_train=None,): 
    model.eval()
    query_eval_loader = DataLoader(eval_dataset, collate_fn=collate_text_val, batch_size=opt.eval_query_bsz,
                                   num_workers=opt.num_workers, shuffle=False, pin_memory=opt.pin_memory)
                    

    if 1: # 下面要返回的list参数们的初始化
        query_metas = []
        query_metas_raw = []

        clip_scale_scores = []
        frame_scale_scores = []
        VT_score_all = []
        TT_score_all = []
        TT_score_global_all = []
        best_clip_indices = []
        score_sum = []

    for idx, batch in tqdm(enumerate(query_eval_loader), desc="Computing q embedding", total=len(query_eval_loader)):
        target, words_mask, idxs, cap_ids, raw_cap_ids,text_real_text = batch
        query_metas.extend(cap_ids) # ['F7TG5', '9JZO2', '9JZO2', '4J1AP',...]
        query_metas_raw.extend(raw_cap_ids) # ['F7TG5#enc#3', '9JZO2#enc#3', '9JZO2#enc#10', '4J1AP#enc#1',...]
        query_feat = target.to(opt.device) # [bsz,max_token_num,1024]
        query_mask = words_mask.to(opt.device) # [bsz,max_token_num]

        normalize_query_feat_norm,TT_score,TT_score_global = model.get_query_global_cap_scores(
            query_feat, query_mask,ctx_info["global_caption_feat"],ctx_info["labels"],query_metas_raw,ctx_info["video_metas"],local_weight)

        _clip_scale_scores, _frame_scale_scores, key_clip_indices = model.get_pred_from_raw_query(
                query_feat, query_mask, None, ctx_info["video_proposal_feat"], ctx_info["video_feat"], ctx_info['video_mask'],)#cross=True
        
        #  ipdb.set_trace()
        if top_k != 0 or top_k == 1:
            if idx == 0:
                print("\n!2 stage!\n")
            TT_score,topk_indices = add_value_to_top_indices(TT_score,top_k)

        _score_sum = opt.clip_scale_w*_clip_scale_scores + opt.frame_scale_w*_frame_scale_scores + TT_weight * TT_score
        VT_score = opt.clip_scale_w*_clip_scale_scores + opt.frame_scale_w*_frame_scale_scores
        #合并batch
        clip_scale_scores.append(_clip_scale_scores.cpu().numpy())
        frame_scale_scores.append(_frame_scale_scores.cpu().numpy())
        VT_score_all.append(VT_score.cpu().numpy())
        TT_score_all.append(TT_score.cpu().numpy())
        score_sum.append(_score_sum.cpu().numpy())
        best_clip_indices.append(key_clip_indices.cpu().numpy()) # [query_num_in_a_batch,video_nums_all]
        
        # 清除缓存
        del _clip_scale_scores, _frame_scale_scores, TT_score, TT_score_global,key_clip_indices
        torch.cuda.empty_cache()  # 可选，根据实际情况判断是否需要

    return (# 直接拼接 NumPy 数组
        np.concatenate(clip_scale_scores, axis=0),  
        np.concatenate(frame_scale_scores, axis=0),
        np.concatenate(VT_score_all, axis=0),
        np.concatenate(TT_score_all, axis=0),
        np.concatenate(score_sum, axis=0),
        np.concatenate(best_clip_indices, axis=0),
        query_metas,
        query_metas_raw
    )


def cal_perf(t2v_all_errors, t2v_gt):
    # video retrieval
    
    (t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr), gt_ranks = eval_q2m(t2v_all_errors, t2v_gt)
    t2v_map_score = t2v_map(t2v_all_errors, t2v_gt)

    logging.info(" * Text to Video:")
    logging.info(" * r_1_5_10_100, medr, meanr: {}".format([round(t2v_r1, 1), round(t2v_r5, 1), round(t2v_r10, 1), round(t2v_r100, 1)]))
    logging.info(" * recall sum: {}".format(round(t2v_r1+t2v_r5+t2v_r10+t2v_r100, 1)))
    logging.info(" * mAP: {}".format(round(t2v_map_score, 4)))
    logging.info(" * "+'-'*10)

    return (t2v_r1, t2v_r5, t2v_r10,t2v_r100, t2v_medr, t2v_meanr, t2v_map_score), gt_ranks

def scores_to_ranks(scores, descending=False):
    """
    将 score 转换为候选级 rank（逐 query）

    Args:
        scores: np.ndarray, shape = (Q, N)
                每一行是一个 query 对所有候选的 score
        descending: bool
                True 表示 score 越大 rank 越靠前（rank=1 最好）

    Returns:
        ranks: np.ndarray, shape = (Q, N)
               ranks[i, j] = 第 i 个 query 下，第 j 个候选的 rank
    """
    Q, N = scores.shape
    ranks = np.zeros_like(scores, dtype=np.int32)

    for i in range(Q):
        if descending:
            ipdb.set_trace()
            order = np.argsort(-scores[i])  # 从大到小
        else:
            order = np.argsort(scores[i])   # 从小到大

        # 把排序位置映射回原索引
        # order[0] -> rank 1
        # order[1] -> rank 2
        ranks[i, order] = np.arange(1, N + 1)
    
    return ranks

def rank_product(VT_score, TT_score, alpha=1.0, beta=1.0):
    VT_rank = scores_to_ranks(VT_score, descending=False)
    TT_rank = scores_to_ranks(TT_score, descending=False)
    return (VT_rank.astype(float) ** alpha) * (TT_rank.astype(float) ** beta)

def rank_product_topk(VT_score, TT_score, K=1000):
    def clip_rank(rank, K):
        """
        将 rank 截断到 Top-K 范围内
        rank > K 的，统一设为 K+1
        """
        return np.minimum(rank, K + 1)

    VT_rank = scores_to_ranks(VT_score, descending=False)
    TT_rank = scores_to_ranks(TT_score, descending=False)

    VT_rank = clip_rank(VT_rank, K)
    TT_rank = clip_rank(TT_rank, K)

    return VT_rank * TT_rank

def eval_epoch(model, val_video_dataset, val_text_dataset, opt):
    # ipdb.set_trace()
    model.eval()
    context_info_train = None
    train_text_dataset = None
    context_info = compute_context_info(model, val_video_dataset, opt) # 提取测试集的video feature (clip, frame) 返回字典
    video_metas = context_info['video_metas']

    if 0: # rank product test
        top_k=1
        local_weight = 0.7
        TT_weight = 0.5
        query_context_scores, frame_scale_scores,VT_score,TT_score, score_sum,best_clip_indices,query_metas, query_metas_raw = \
        compute_query2ctx_info(model, val_text_dataset, opt, context_info,TT_weight,top_k,local_weight,train_text_dataset,context_info_train)
        logger.info(f"\nTT_weight:{TT_weight:.2f} top_k:{top_k:.2f} local_weight:{local_weight:.2f}")
        v2t_gt, t2v_gt = get_gt(video_metas, query_metas)

        if 0: # 5:4 80.0
            for alpha in np.arange(0.3, 1.6, 0.3):
                logger.info(f"\n alpha: {alpha}")
                for beta in np.arange(0.3, 1.6, 0.3):
                    RP = rank_product(VT_score, TT_score, alpha, beta)
                    (t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr, t2v_map_score), gt_ranks = cal_perf(-1 * RP, t2v_gt)
        if 1: # topk is 900 79.7
            for k in range(300,1501,200): 
                RP = rank_product_topk(VT_score, TT_score, k)
                (t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr, t2v_map_score), gt_ranks = cal_perf(-1 * RP, t2v_gt)
        # (t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr, t2v_map_score), gt_ranks = cal_perf(-1 * TT_score, t2v_gt)
        # (t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr, t2v_map_score), gt_ranks = cal_perf(-1 * score_sum, t2v_gt)

    if 0: # local weight and TT weight
        logger.info(f"local weight超参 bs+TT")
        # TT_weight=0.7
        top_k=1
        for local_weight in np.arange(0.3,1.0,1):
            for TT_weight in np.arange(0.7,1.0,1):
                query_context_scores, frame_scale_scores,VT_score,TT_score, score_sum,best_clip_indices,query_metas, query_metas_raw = \
                    compute_query2ctx_info(model, val_text_dataset, opt, context_info,TT_weight,top_k,local_weight,train_text_dataset,context_info_train)
                logger.info(f"\nTT_weight:{TT_weight:.2f} top_k:{top_k:.2f} local_weight:{local_weight:.2f}")
                v2t_gt, t2v_gt = get_gt(video_metas, query_metas)
                (t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr, t2v_map_score), gt_ranks = cal_perf(-1 * TT_score, t2v_gt)
                (t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr, t2v_map_score), gt_ranks = cal_perf(-1 * score_sum, t2v_gt)

    if 0: # TT_weight 超参
        logger.info(f"TT weight超参 bs+TT")
        local_weight = 0.3 # best local weight 103.2
        top_k = 0
        for TT_weight in np.arange(0.0,1.1,0.1):
            local_weight = 0.3
            TT_weight = 0.7
            query_context_scores, frame_scale_scores,VT_score,TT_score, score_sum,best_clip_indices,query_metas, query_metas_raw = \
                compute_query2ctx_info(model, val_text_dataset, opt, context_info,TT_weight,top_k,local_weight,train_text_dataset,context_info_train)
            logger.info(f"\nTT_weight:{TT_weight:.2f} top_k:{top_k:.2f} local_weight:{local_weight:.2f}")
            v2t_gt, t2v_gt = get_gt(video_metas, query_metas) # t2v_gt: text_index -> video_index ['3MSZA', '3MSZA', '3MSZA', '3MSZA', 'AMT7R'] -> 0: [0], 1: [0], 2: [0], 3: [0], 4: [1]
            (t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr, t2v_map_score), gt_ranks = cal_perf(-1 * score_sum, t2v_gt)
    
    if 1: # 2 stage top_k超参
        logger.info(f"TT实验——TT top_k超参 TT")
        TT_weight = 0.7
        local_weight = 0.3
        for top_k in np.arange(1):
            top_k = 1
            query_context_scores, frame_scale_scores,VT_score,TT_score, score_sum,best_clip_indices,query_metas, query_metas_raw = \
                compute_query2ctx_info(model, val_text_dataset, opt, context_info,TT_weight,top_k,local_weight,train_text_dataset,context_info_train)
            logger.info(f"\nTT_weight:{TT_weight:.2f} top_k:{top_k:.2f} local_weight:{local_weight:.2f}")
            v2t_gt, t2v_gt = get_gt(video_metas, query_metas)
            (t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr, t2v_map_score), gt_ranks = cal_perf(-1 * score_sum, t2v_gt)
            (t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr, t2v_map_score), gt_ranks = cal_perf(-1 * TT_score, t2v_gt)

    
    currscore = 0
    return currscore


def setup_model(opt):
    """Load model from checkpoint and move to specified device"""
    ckpt_filepath = os.path.join(opt.ckpt_filepath)
    # checkpoint = torch.load(ckpt_filepath)
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
    # caption_files = {x: os.path.join(rootpath, collection, 'TextData', cap_file[x])
    #                  for x in cap_file}

    text_feat_path = '../../depends/generate_caption/youcook2/extracted_data/youcook2_query_roberta.hdf5'
    # text_feat_path = os.path.join(rootpath, collection, 'TextData', 'roberta_%s_query_feat.hdf5' % collection)
    # Load visual features
    # visual_feat_path = os.path.join(rootpath, collection, 'FeatureData', opt.visual_feature)

    # visual_feats = BigFile(visual_feat_path)
    # video2frames =  read_dict(os.path.join(rootpath, collection, 'FeatureData', opt.visual_feature, 'video2frames.txt'))

    visual_feats = None
    video2frames = None
    test_video_ids_list = read_video_ids('../../depends/generate_caption/youcook2/extracted_data/validation.txt')
    test_vid_dataset = VisDataSet4MS_SL(visual_feats, video2frames, opt,
                                               video_ids=test_video_ids_list)
    test_text_dataset = TxtDataSet4MS_SL('../../depends/generate_caption/youcook2/extracted_data/validation.txt', text_feat_path, opt)
    # global_caption



    model = setup_model(opt)

    logger.info("Starting inference...")
    with torch.no_grad():
        score = eval_epoch(model, test_vid_dataset, test_text_dataset, opt)



if __name__ == '__main__':
    start_inference()