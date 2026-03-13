import json
import torch
import torch.utils.data as data
import numpy as np
import re
import h5py
import random
import math
import os
# from transformers import RobertaTokenizer, RobertaModel
import time        
import collections

# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)

def read_json(file_path):
    """
    读取 JSON 文件并返回字典对象。
    
    :param file_path: JSON 文件的路径
    :return: 返回字典对象
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
def is_tensor_with_single_dimension(var):
    if isinstance(var, torch.Tensor):
        return len(var.shape) == 1
    return False
def getVideoId(cap_id):
    vid_id = cap_id.split('#')[0]
    return vid_id

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    return string.strip().lower().split()

def read_video_ids(cap_file):
    video_ids_list = []
    with open(cap_file, 'r') as cap_reader:
        for line in cap_reader.readlines():
            cap_id, caption = line.strip().split(' ', 1)
            video_id = getVideoId(cap_id)
            if video_id not in video_ids_list:
                video_ids_list.append(video_id)
    return video_ids_list

def average_to_fixed_length(visual_input, map_size):
    visual_input = torch.from_numpy(visual_input)
    num_sample_clips = map_size
    num_clips = visual_input.shape[0]
    idxs = torch.arange(0, num_sample_clips + 1, 1.0) / num_sample_clips * num_clips

    idxs = torch.min(torch.round(idxs).long(), torch.tensor(num_clips - 1))

    new_visual_input = []

    for i in range(num_sample_clips):

        s_idx, e_idx = idxs[i].item(), idxs[i + 1].item()
        if s_idx < e_idx:
            new_visual_input.append(torch.mean(visual_input[s_idx:e_idx], dim=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0).numpy()

    return new_visual_input

def uniform_feature_sampling_wrong(features, max_len):
    num_clips = features.shape[0]
    if max_len is None or num_clips <= max_len:
        return features
    idxs = np.arange(0, max_len + 1, 1.0) / max_len * num_clips
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs > num_clips - 1] = num_clips - 1
    # 计算采样索引
    # idxs = np.linspace(0, num_clips - 1, max_len + 1, dtype=np.int32)

    # 使用列表推导和预先计算的区间来计算平均值
    #start和end如果相等就会出错
    new_features = np.array([np.mean(features[start:end], axis=0)for start, end in zip(idxs[:-1], idxs[1:])])

    return new_features

def uniform_feature_sampling(features, max_len):
    num_clips = features.shape[0]
    if max_len is None or num_clips <= max_len:
        return features
    idxs = np.arange(0, max_len + 1, 1.0) / max_len * num_clips
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs > num_clips - 1] = num_clips - 1
    new_features = []
    for i in range(max_len):
        s_idx, e_idx = idxs[i], idxs[i + 1]
        if s_idx < e_idx:
            new_features.append(np.mean(features[s_idx:e_idx], axis=0))
        else:
            new_features.append(features[s_idx])
    new_features = np.asarray(new_features)
    return new_features

def optimized_uniform_feature_sampling(features, max_len):
    num_clips = features.shape[0]
    if max_len is None or num_clips <= max_len:
        return features
    
    # Generate indices once and use vectorized operations
    idxs = np.linspace(0, num_clips - 1, max_len + 1, dtype=np.int32)
    
    # Compute all slices at once
    slices = [features[s_idx:e_idx] for s_idx, e_idx in zip(idxs[:-1], idxs[1:])]
    
    # Compute means using vectorized operation
    new_features = np.array([np.mean(slice, axis=0) if slice.shape[0] > 1 else slice[0] for slice in slices])
    
    return new_features

def l2_normalize_np_array(np_array, eps=1e-5):
    """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)

def l2_normalize_tensor(tensor, eps=1e-5):
    """tensor: torch.Tensor, (*, D), where the last dim will be normalized"""
    return tensor / (torch.norm(tensor, dim=-1, keepdim=True) + eps)

def collate_train(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # Sort a data list by caption length
    if data[0][1] is not None:
        data.sort(key=lambda x: len(x[1]), reverse=True)
    clip_video_features, frame_video_features, captions, idxs, cap_ids, video_ids, global_caption_mask, local_start_end_tensor = zip(*data)

    #videos
    clip_videos = torch.cat(clip_video_features, dim=0).float()

    video_lengths = [len(frame) for frame in frame_video_features]
    frame_vec_len = len(frame_video_features[0][0])
    frame_videos = torch.zeros(len(frame_video_features), max(video_lengths), frame_vec_len)
    videos_mask = torch.zeros(len(frame_video_features), max(video_lengths))
    for i, frames in enumerate(frame_video_features):
        end = video_lengths[i]
        frame_videos[i, :end, :] = frames[:end, :]
        videos_mask[i, :end] = 1.0

    #captions
    feat_dim = captions[0][0].shape[-1]

    merge_captions = []
    all_lengths = []
    labels = []

    for index, caps in enumerate(captions):
        labels.extend(index for i in range(len(caps)))
        all_lengths.extend(len(cap) for cap in caps)
        merge_captions.extend(cap for cap in caps)

    target = torch.zeros(len(all_lengths), max(all_lengths), feat_dim)
    words_mask = torch.zeros(len(all_lengths), max(all_lengths))

    for index, cap in enumerate(merge_captions):
        end = all_lengths[index]
        target[index, :end, :] = cap[:end, :]
        words_mask[index, :end] = 1.0

    # 展开 global_caption_mask，顺序与 merge_captions 一致
    merge_global_caption_mask = []
    for masks in global_caption_mask:
        merge_global_caption_mask.extend(masks)
    
    # 展平 local_start_end_tensor，顺序与 merge_captions 一致
    merge_local_start_end_tensor = []
    for tensors in local_start_end_tensor:
        merge_local_start_end_tensor.extend(tensors)
    merge_local_start_end_tensor = torch.stack(merge_local_start_end_tensor, dim=0)
    
    return dict(clip_video_features=clip_videos,
                frame_video_features=frame_videos,
                videos_mask=videos_mask,
                text_feat=target,
                text_mask=words_mask,
                text_labels=labels,
                global_caption_mask=merge_global_caption_mask,
                local_start_end_tensor=merge_local_start_end_tensor
                )

def collate_frame_val(data,need_caption = True):
    if 1:
        clip_video_features, frame_video_features, idxs, video_ids,global_captions,start_end_list = zip(*data)
    else:
        clip_video_features, frame_video_features, idxs, video_ids = zip(*data)
    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    # videos

    clip_videos = torch.cat(clip_video_features, dim=0).float()
    
    video_lengths = [len(frame) for frame in frame_video_features]
    frame_vec_len = len(frame_video_features[0][0])
    frame_videos = torch.zeros(len(frame_video_features), max(video_lengths), frame_vec_len)
    videos_mask = torch.zeros(len(frame_video_features), max(video_lengths))
    for i, frames in enumerate(frame_video_features):
        end = video_lengths[i]
        frame_videos[i, :end, :] = frames[:end, :]
        videos_mask[i, :end] = 1.0

    # captions
    if 1:
        feat_dim = global_captions[0][0].shape[-1]
        merge_captions = []
        all_lengths = []
        labels = []

        for index, caps in enumerate(global_captions):
            labels.extend(idxs[index] for i in range(len(caps)))
            all_lengths.extend(len(cap) for cap in caps)
            merge_captions.extend(cap for cap in caps)

        padding_global_caption = torch.zeros(len(all_lengths), max(all_lengths), feat_dim)
        global_caption_mask = torch.zeros(len(all_lengths), max(all_lengths))
        
        for index, cap in enumerate(merge_captions):
            end = all_lengths[index]
            padding_global_caption[index, :end, :] = cap[:end, :]
            global_caption_mask[index, :end] = 1.0

        start_end_flattened = [item for sublist in start_end_list for item in sublist]
        # import ipdb;ipdb.set_trace()
        return clip_videos, frame_videos, videos_mask, idxs, labels, video_ids,padding_global_caption,global_caption_mask,start_end_flattened

    # return clip_videos, frame_videos, videos_mask, idxs, labels, video_ids

def collate_text_val(data,test_dataset = False):
    all_captions = []
    all_cap_ids = []
    all_raw_cap_ids = []
    all_idxs = []
    all_real_text = []
    # 拉平成一个 caption 列表

    for i, (cap_list, idx, cap_id, raw_cap_id, real_text) in enumerate(data):
        if isinstance(cap_list, torch.Tensor):
            cap_list = [cap_list]  # 兼容旧版本
        for cap in cap_list:
            all_captions.append(cap)
            all_cap_ids.append(cap_id)
            all_raw_cap_ids.append(raw_cap_id)
            all_idxs.append(idx)
        all_real_text.extend(real_text)

    # 计算每个 caption 的长度
    cap_lens = [len(cap) for cap in all_captions]
    max_len = max(cap_lens)
    feat_dim = all_captions[0].shape[-1]

    # 构建 target 和 mask
    target = torch.zeros(len(all_captions), max_len, feat_dim)
    words_mask = torch.zeros(len(all_captions), max_len)

    for i, cap in enumerate(all_captions):
        cur_len = len(cap)
        target[i, :cur_len] = cap
        words_mask[i, :cur_len] = 1.0

    return target, words_mask, all_idxs, all_cap_ids, all_raw_cap_ids,all_real_text

    # (batch_size, max_seq_len, feature_dim),(batch_size, max_seq_len)
    # all_idxs = [0, 0, 1, 2, 2, 2] 0号视频的queries
    # AO8RW
    # AO8RW#0



class Dataset4MS_SL(data.Dataset):
    """
    Load captions and video frame features by pre-trained CNN model.
    """

    def __init__(self, cap_file, visual_feat, text_feat_path, opt, video2frames=None):
        # Captions
        self.captions = {}
        self.cap_ids = []
        self.video_ids = []
        self.vid_caps = {}
        self.video2frames = video2frames

        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.split(' ', 1)
                video_id = getVideoId(cap_id)
                self.captions[cap_id] = caption.strip()
                self.cap_ids.append(cap_id)
                if video_id not in self.video_ids:
                    self.video_ids.append(video_id)
                if video_id in self.vid_caps:
                    self.vid_caps[video_id].append(cap_id)
                else:
                    self.vid_caps[video_id] = []
                    self.vid_caps[video_id].append(cap_id)
        self.visual_feat = visual_feat
        self.text_feat_path = text_feat_path
        self.map_size = opt.map_size
        self.max_ctx_len = opt.max_ctx_l
        self.max_desc_len = opt.max_desc_l

        self.length = len(self.vid_caps)
        self.config = opt
        
        if opt.dset_name =='activitynet':
            self.video_feat_path = '../../depends/activitynet/anet_clip_i3d_numpy.hdf5'
        elif opt.dset_name =='tvr':
            self.video_feat_path = "../../depends/tvr/tvr_clip_i3d_numpy.hdf5"
        elif opt.dset_name =='charades':
            self.video_feat_path = '../../depends/charades/charades_clip_i3d_numpy.hdf5'
        elif opt.dset_name =='youcook2':
            self.video_feat_path = '../../depends/youcook2/Youcook2/video_i3d_feats.h5'
          
        self.video_feat_file = h5py.File(self.video_feat_path, 'r')
        
        self.query_feat_file = h5py.File(self.text_feat_path, 'r')

        if self.config.caption_rate > 0:
            if opt.dset_name =='charades':
                self.query_feat_file_caption_gpt = '../../depends/charades/charades_local_all_paraphrased.hdf5'
            elif opt.dset_name =='activitynet':
                self.query_feat_file_caption_gpt = '../../depends/activitynet/anet_local_all.hdf5'
            elif opt.dset_name =='youcook2':
                self.query_feat_file_caption_gpt = '../../depends/youcook2/youcook_train_020.hdf5'

            self.query_feat_file_caption_from_gpt = h5py.File(self.query_feat_file_caption_gpt, 'r')

            # 顺序获取所有key
            all_keys = list(self.query_feat_file_caption_from_gpt.keys())

            # 截取caption_rate的key
            num_total = len(all_keys)
            num_selected = int(self.config.caption_rate * num_total)
            selected_keys = all_keys[:num_selected] # 顺序读取
            # selected_keys = all_keys[-num_selected:] # 逆序读取


            # 映射成字典 {video_id: [caption_id1, caption_id2, ...]}
            self.caption_gpt_map = {}
            for key in selected_keys:
                if opt.dset_name == 'charades':
                    video_id = key.split('_')[0] # 不要后半部分的起止时间
                elif opt.dset_name == 'activitynet':
                    video_id = key.rsplit('_', 1)[0]
                if video_id not in self.caption_gpt_map:
                    self.caption_gpt_map[video_id] = []
                self.caption_gpt_map[video_id].append(key) # 用append防止替换而不是添加
            
            print(f"[扩增] 选取前 {self.config.caption_rate * 100:.1f}%：共 {num_selected} 条，{len(self.caption_gpt_map)} 个视频。")

        # global_caption 处理，过程类似local caption
        if hasattr(self.config, 'global_caption') and self.config.global_caption:
            if opt.dset_name =='charades':
                self.query_feat_file_global_caption = '../../depends/charades/charades_global_qwen32b.hdf5'
            elif opt.dset_name =='activitynet':
                self.query_feat_file_global_caption = '../../depends/activitynet/anet_global_roberta.hdf5'
                
            self.query_feat_file_global_caption_from_gpt = h5py.File(self.query_feat_file_global_caption, 'r')
            all_keys = list(self.query_feat_file_global_caption_from_gpt.keys())
            self.global_caption_map = {}
            for key in all_keys:
                if opt.dset_name == 'charades':
                    video_id = key.split('_')[0]
                elif opt.dset_name == 'activitynet':
                    video_id = key.rsplit('_', 1)[0]
                if video_id not in self.global_caption_map:
                    self.global_caption_map[video_id] = []
                self.global_caption_map[video_id].append(key)
            print(f"[全局caption] 共 {len(all_keys)} 条，{len(self.global_caption_map)} 个视频。")
    

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        cap_ids = self.vid_caps[video_id]

        if self.config.dset_name in {'activitynet', 'charades', 'tvr'}:
            temp_feat = self.video_feat_file[video_id]['i3d_feat'][...]
            clip_video_feature = average_to_fixed_length(temp_feat, self.map_size)
            clip_video_feature = torch.from_numpy(l2_normalize_np_array(clip_video_feature)).unsqueeze(0)

            frame_video_feature = uniform_feature_sampling(temp_feat, self.max_ctx_len)
            frame_video_feature = torch.from_numpy(l2_normalize_np_array(frame_video_feature))

        cap_tensors = [] # 建立一个新的数组存储添加后的caption
        global_caption_mask = []  # 新增：记录global caption的位置，1为global caption，0为其他
        local_start_end_tensor = []  # 新增：记录local/global/original的起止时间

        # 添加global caption（如果有）
        if hasattr(self, 'global_caption_map') and video_id in self.global_caption_map:
            temp_global_caption_list = self.global_caption_map[video_id]
            for temp_global_cap_id in temp_global_caption_list:
                cap_feat = self.query_feat_file_global_caption_from_gpt[temp_global_cap_id][...]
                cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]
                cap_tensors.append(cap_tensor)
                global_caption_mask.append(1)  # 1表示global caption
                local_start_end_tensor.append(torch.tensor([0.0, 0.0]))  # global caption 起止时间为[0,0]

        # 加入我们提取的文本特征
        if self.config.caption_rate > 0.0 and video_id in self.caption_gpt_map:
            temp_caption_list = self.caption_gpt_map[video_id]  # 比如 ['I121D_0.20-0.50', 'I121D_0.50-0.80']
            for temp_cap_id in temp_caption_list:
                cap_feat = self.query_feat_file_caption_from_gpt[temp_cap_id][...]
                cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]
                cap_tensors.append(cap_tensor)
                global_caption_mask.append(2)  # 2表示local caption
                # 解析local caption的起止时间
                try:
                    # 取最后一个下划线后的内容
                    time_str = temp_cap_id.rsplit('_', 1)[-1]  # 例如 '0.20-0.50'
                    start_str, end_str = time_str.split('-')
                    start = float(start_str)
                    end = float(end_str)
                    local_start_end_tensor.append(torch.tensor([start, end]))
                except Exception as e:
                    raise RuntimeError(f"local caption id '{temp_cap_id}' 起止时间解析失败: {e}")

        # 原始标注的caption
        for cap_id in cap_ids:
            cap_feat = self.query_feat_file[cap_id][...]
            cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]
            cap_tensors.append(cap_tensor)
            global_caption_mask.append(0)  # 0表示query
            local_start_end_tensor.append(torch.tensor([0.0, 0.0]))  # 原始caption起止时间为[0,0]

        # 4. 最后统一筛选 local caption
        local_caption_keep_num = self.config.local_caption_keep_num
        start_end_to_indices = collections.defaultdict(list)
        for idx, (mask, se) in enumerate(zip(global_caption_mask, local_start_end_tensor)):
            if mask == 2:
                start_end_tuple = tuple(se.tolist())
                start_end_to_indices[start_end_tuple].append(idx)
        keep_indices = set()
        for indices in start_end_to_indices.values():
            n_keep = min(local_caption_keep_num, len(indices))
            keep = random.sample(indices, n_keep)
            keep_indices.update(keep)
        new_cap_tensors = []
        new_global_caption_mask = []
        new_local_start_end_tensor = []
        for i, (cap, mask, se) in enumerate(zip(cap_tensors, global_caption_mask, local_start_end_tensor)):
            if mask != 2 or i in keep_indices:
                new_cap_tensors.append(cap)
                new_global_caption_mask.append(mask)
                new_local_start_end_tensor.append(se)

        return clip_video_feature, frame_video_feature, new_cap_tensors, index, cap_ids, video_id, new_global_caption_mask, new_local_start_end_tensor

    
    def __len__(self):
        return self.length
    
class VisDataSet4MS_SL(data.Dataset): # need_caption为真是测试集，取test captions，反之取训练集的captions 这里没有query

    def __init__(self, visual_feat, video2frames, opt, video_ids=None,need_caption = True):
        self.visual_feat = visual_feat
        self.video2frames = video2frames
        if video_ids is not None:
            self.video_ids = video_ids
        else:
            self.video_ids = video2frames.keys()
        self.length = len(self.video_ids)
        self.map_size = opt.map_size
        self.max_ctx_len = opt.max_ctx_l
        self.max_desc_len = opt.max_desc_l
        self.config = opt
        self.need_caption = need_caption
        
        if opt.dset_name =='activitynet':
            self.video_feat_path = '../../depends/activitynet/anet_clip_i3d_numpy.hdf5'
            if self.need_caption: # 测试集需要test caption 
                self.global_caption_path = '../../depends/activitynet/anet_test_global.hdf5'
                self.local_caption_path = '../../depends/activitynet/anet_test_020_merged.hdf5'

        elif opt.dset_name =='tvr':
            self.video_feat_path = '../../depends/tvr/tvr_clip_i3d_numpy.hdf5'
        elif opt.dset_name =='charades':
            self.video_feat_path = '../../depends/charades/charades_clip_i3d_numpy.hdf5'
            self.global_caption_path = '../../depends/charades/charades_global_test.hdf5' # 85.1
            self.local_caption_path = '../../depends/charades/charades_test_020_merged.hdf5' # 7/10生成的
                
        elif opt.dset_name =='youcook2':
            self.video_feat_path = '../../depends/youcook2/video_i3d_feats.h5'
            self.local_caption_path = '../../depends/youcook2/youcook_val_020_merged.hdf5'
            self.global_caption_path = '../../depends/youcook2/youcook_val_global.hdf5'
        self.global_caption_file = h5py.File(self.global_caption_path, 'r')
        
        self.local_caption_file = h5py.File(self.local_caption_path, 'r')
        all_keys = list(self.global_caption_file.keys())
        self.global_caption_map = {}
        for key in all_keys:
            if opt.dset_name == 'charades':
                video_id = key.split('_')[0]
                # print(key, video_id)
            elif opt.dset_name == 'activitynet' or opt.dset_name == 'youcook2':
                video_id = key.rsplit('_', 1)[0]
            if video_id not in self.global_caption_map:
                self.global_caption_map[video_id] = []
            self.global_caption_map[video_id].append(key) # 'ZYJJF': ['ZYJJF_0.00-1.00']
        print(f"global captions in test set: {len(all_keys)}")

        local_all_keys = list(self.local_caption_file.keys())
        self.local_caption_map = {}
        for key in local_all_keys:
            if opt.dset_name == 'charades':
                video_id = key.split('_')[0]
            elif opt.dset_name == 'activitynet' or opt.dset_name == 'youcook2':
                video_id = key.rsplit('_', 1)[0]
            if video_id not in self.local_caption_map:
                self.local_caption_map[video_id] = []
            self.local_caption_map[video_id].append(key) # 'ZYJJF': ['ZYJJF_0.00-1.00']
        print(f"local captions in test set: {len(local_all_keys)}")

        self.video_feat_file = h5py.File(self.video_feat_path, 'r')
            
    def __getitem__(self, index):
        video_id = self.video_ids[index]
        start_end_list = []
        if self.config.dset_name=='activitynet' or self.config.dset_name=='charades' or self.config.dset_name=='tvr' or self.config.dset_name=='youcook2':
            temp_feat = self.video_feat_file[video_id]['i3d_feat'][...]
            clip_video_feature = average_to_fixed_length(temp_feat, self.map_size)
            clip_video_feature = torch.from_numpy(l2_normalize_np_array(clip_video_feature)).unsqueeze(0)

            frame_video_feature = uniform_feature_sampling(temp_feat, self.max_ctx_len)
            frame_video_feature = torch.from_numpy(l2_normalize_np_array(frame_video_feature))#.unsqueeze(0)

            if 1:
                cap_ids = self.global_caption_map[video_id]
                cap_ids_local = self.local_caption_map[video_id]
                # import ipdb;ipdb.set_trace()
                cap_tensors = []            
                for cap_id in cap_ids_local:
                    start = float(cap_id.rsplit('_', 1)[1].rsplit('-', 1)[0])
                    end = float(cap_id.rsplit('_', 1)[1].rsplit('-', 1)[1])
                    start_end_list.append((start,end))

                    cap_feat_local = self.local_caption_file[cap_id][...]
                    cap_tensor_local = torch.from_numpy(l2_normalize_np_array(cap_feat_local))[:self.max_desc_len]
                    cap_tensors.append(cap_tensor_local)
                    
                for cap_id in cap_ids:
                    start = float(cap_id.rsplit('_', 1)[1].rsplit('-', 1)[0])
                    end = float(cap_id.rsplit('_', 1)[1].rsplit('-', 1)[1])
                    start_end_list.append((start,end))

                    cap_feat = self.global_caption_file[cap_id][...]
                    cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]
                    cap_tensors.append(cap_tensor)
                return clip_video_feature, frame_video_feature, index, video_id,cap_tensors,start_end_list

        else:
            frame_video_feature = uniform_feature_sampling(np.load(save_path), self.max_ctx_len)
            frame_video_feature = l2_normalize_np_array(frame_video_feature)
            frame_video_feature = torch.from_numpy(frame_video_feature)         
        return clip_video_feature, frame_video_feature, index, video_id

    def __len__(self):
        # return 20
        return self.length

class TxtDataSet4MS_SL(data.Dataset): # 得到test query
    """
    Load captions
    """

    def __init__(self, cap_file, text_feat_path, opt):
        # Captions
        self.captions = {}
        self.cap_ids = []
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
        self.text_feat_path = text_feat_path
        self.max_desc_len = opt.max_desc_l
        self.length = len(self.cap_ids)

        self.map_size = opt.map_size
        self.text_feat = h5py.File(self.text_feat_path, 'r')
    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        cap_feat = self.text_feat[cap_id][...].squeeze()
        cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]
        caption=[]
        caption.append(self.captions[cap_id])

        # 一次返回一条，而不是一个视频的
        return cap_tensor, index, cap_id.split('#')[0], cap_id,caption

    def __len__(self):
        return self.length

class TxtDataSet4MS_SL_train(data.Dataset): # 得到 train query+captions 用于VV中的TTV环节
    """
    Load query 删掉VV后 这里仅有query
    """

    def __init__(self, cap_file, text_feat_path, opt):
        # Captions
        self.captions = {} # 这个是text
        self.cap_ids = []
        self.video_ids = []
        self.video2cap={}
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                self.captions[cap_id] = caption

                self.cap_ids.append(cap_id)
                video_id = cap_id.split('#')[0]
                if video_id not in self.video_ids:
                    self.video_ids.append(video_id)
                    self.video2cap[video_id] = []
                self.video2cap[video_id].append(cap_id)

        self.text_feat_path = text_feat_path
        self.max_desc_len = opt.max_desc_l
        self.length = len(self.video_ids)
        self.config = opt

        self.map_size = opt.map_size
        self.text_feat = h5py.File(self.text_feat_path, 'r')

        if self.config.caption_rate > 0:
            if opt.dset_name =='charades':
                self.query_feat_file_caption_gpt_text = '../../depends/charades/LastVers/charades_train_local_2scales0205_merged.json' # 7/7生成的multi scale去掉0.2
                self.query_feat_file_caption_gpt = '../../depends/charades/charades_train_020_merged.hdf5' # 7/10生成的
                with open(self.query_feat_file_caption_gpt_text, 'r') as f:
                    self.local_caption_text = json.load(f)
                    self.local_caption_text_map = {}
                    for video_id, caption_list in self.local_caption_text.items():
                        for item in caption_list:
                            start = item["start_time"]
                            end = item["end_time"]
                            caption = item["caption"]
                            key = f"{video_id}_{start:.2f}-{end:.2f}"
                            self.local_caption_text_map[key] = caption

            elif opt.dset_name =='activitynet' or opt.dset_name == 'youcook2':
                exit()
            self.query_feat_file_caption_from_gpt = h5py.File(self.query_feat_file_caption_gpt, 'r')
            
            # 顺序获取所有key
            all_keys = list(self.query_feat_file_caption_from_gpt.keys())

            # 截取caption_rate的key
            num_total = len(all_keys)
            num_selected = int(self.config.caption_rate * num_total)
            selected_keys = all_keys[:num_selected] # 顺序读取

            # 映射成字典 {video_id: [caption_id1, caption_id2, ...]}
            self.caption_gpt_map = {}
            for key in selected_keys:
                if opt.dset_name == 'charades' or opt.dset_name == 'youcook2':
                    video_id = key.split('_')[0] # 不要后半部分的起止时间
                elif opt.dset_name == 'activitynet':
                    video_id = key.rsplit('_', 1)[0]
                if video_id not in self.caption_gpt_map:
                    self.caption_gpt_map[video_id] = []
                self.caption_gpt_map[video_id].append(key) # 用append防止替换而不是添加
            
            print(f"train in eval [local] 选取前 {self.config.caption_rate * 100:.1f}%：共 {num_selected} 条，{len(self.caption_gpt_map)} 个视频。")

        # global_caption 处理，过程类似local caption
        if hasattr(self.config, 'global_caption') and self.config.global_caption:
            import ipdb;ipdb.set_trace()
            if opt.dset_name =='charades':
                self.query_feat_file_global_caption = '../../depends/charades/charades_global_qwen32b.hdf5' # 74.5的
                self.query_feat_file_global_caption_text = '../../depends/charades/charades_qwenvl_7b_caption_global_train_2fps.json' # 74.5的
                with open(self.query_feat_file_global_caption_text, 'r') as f:
                    self.global_caption_text = json.load(f)
                    self.global_caption_text_map = {}
                    for video_id, caption_list in self.global_caption_text.items():
                        for item in caption_list:
                            start = item["start_time"]
                            end = item["end_time"]
                            caption = item["caption"]
                            key = f"{video_id}_{start:.2f}-{end:.2f}"
                            self.global_caption_text_map[key] = caption
            elif opt.dset_name =='activitynet':
                self.query_feat_file_global_caption = '../../depends/activitynet/anet_global_roberta.hdf5'
            elif opt.dset_name =='youcook2':
                self.query_feat_file_global_caption = '../../depends/youcook2/youcook_val_global.hdf5'
                
            self.query_feat_file_global_caption_from_gpt = h5py.File(self.query_feat_file_global_caption, 'r')
            all_keys = list(self.query_feat_file_global_caption_from_gpt.keys())
            self.global_caption_map = {}
            for key in all_keys:
                if opt.dset_name == 'charades' or opt.dset_name == 'youcook2':
                    video_id = key.split('_')[0]
                elif opt.dset_name == 'activitynet':
                    video_id = key.rsplit('_', 1)[0]
                if video_id not in self.global_caption_map:
                    self.global_caption_map[video_id] = []
                self.global_caption_map[video_id].append(key)
            print(f"train in eval [global] 共 {len(all_keys)} 条，{len(self.global_caption_map)} 个视频。")
            # import ipdb;ipdb.set_trace()
    def __getitem__(self, index):
        video_id=self.video_ids[index]
        cap_id_list = self.video2cap[video_id]
        real_text = [] # 按特征读的顺序存text
        cap_tensors = []  # 存储所有的caption feature

        # import ipdb;ipdb.set_trace()
        # 加入 global caption（如果存在）
        if 1:
            if hasattr(self, 'global_caption_map') and video_id in self.global_caption_map:
                global_cap_ids = self.global_caption_map[video_id]
                for gc_id in global_cap_ids:
                    gc_feat = self.query_feat_file_global_caption_from_gpt[gc_id][...]
                    gc_tensor = torch.from_numpy(l2_normalize_np_array(gc_feat))[:self.max_desc_len]
                    cap_tensors.append(gc_tensor)
                    real_text.append(f"{gc_id}: {self.global_caption_text_map[gc_id]}")
            # 加入 local caption（如果存在）
            if self.config.caption_rate > 0.0 and video_id in self.caption_gpt_map:
                local_cap_ids = self.caption_gpt_map[video_id]
                for lc_id in local_cap_ids:
                    lc_feat = self.query_feat_file_caption_from_gpt[lc_id][...]
                    lc_tensor = torch.from_numpy(l2_normalize_np_array(lc_feat))[:self.max_desc_len]
                    cap_tensors.append(lc_tensor)
                    real_text.append(f"{lc_id}: {self.local_caption_text_map[lc_id]}")            

        # 加入原始query
            for cap_id in cap_id_list:
                if 1:
                    cap_feat = self.text_feat[cap_id][...].squeeze()
                    cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]
                    cap_tensors.append(cap_tensor)
                    real_text.append(f"{cap_id}: {self.captions[cap_id]}")

        # import ipdb;ipdb.set_trace()
        return cap_tensors, index, video_id, cap_id, real_text




    def __len__(self):
        return self.length

if __name__ == '__main__':
    pass

"""
VisDataSet4MS_SL 执行两次。分别取出test 的 video feature + captions 和 train 的 video feature + captions
TxtDataSet4MS_SL 执行一次。取出test query
TxtDataSet4MS_SL_train 执行一次 取出train query + captions
TT: test query @ test captions (在compute_query2ctx_info中算的,这两个的计算用这两个: query_feat,ctx_info["global_caption_feat"])
VV: test video (测试集query @ 训练集query + captions) train video 
"""



