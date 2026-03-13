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

def collate_frame_val(data):
    clip_video_features, frame_video_features, idxs, video_ids = zip(*data)  # ,global_captions
    
    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    # videos
    clip_videos = torch.cat(clip_video_features, dim=0).float()

    # #global_captions
    # feat_dim = global_captions[0][0].shape[-1]

    # merge_captions = []
    # all_lengths = []

    # for index, cap in enumerate(global_captions):
    #     all_lengths.append(len(cap)) # 取长度
    #     merge_captions.append(cap) # 取特征

    # padding_global_caption = torch.zeros(len(all_lengths), max(all_lengths), feat_dim)
    # global_caption_mask = torch.zeros(len(all_lengths), max(all_lengths))

    # for index, cap in enumerate(merge_captions):
    #     end = all_lengths[index]
    #     padding_global_caption[index, :end, :] = cap[:end, :]
    #     global_caption_mask[index, :end] = 1.0

    video_lengths = [len(frame) for frame in frame_video_features]
    frame_vec_len = len(frame_video_features[0][0])
    frame_videos = torch.zeros(len(frame_video_features), max(video_lengths), frame_vec_len)
    videos_mask = torch.zeros(len(frame_video_features), max(video_lengths))
    for i, frames in enumerate(frame_video_features):
        end = video_lengths[i]
        frame_videos[i, :end, :] = frames[:end, :]
        videos_mask[i, :end] = 1.0

    return clip_videos, frame_videos, videos_mask, idxs, video_ids # ,padding_global_caption,global_caption_mask

def collate_text_val(data):
    if data[0][0] is not None:
        data.sort(key=lambda x: len(x[0]), reverse=True)
    captions,idxs, cap_ids, raw_cap_ids = zip(*data)

    if captions[0] is not None:
        # Merge captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        target = torch.zeros(len(captions), max(lengths), captions[0].shape[-1])
        words_mask = torch.zeros(len(captions), max(lengths))
        for i, cap in enumerate(captions):
            end = lengths[i]
            target[i, :end] = cap[:end]
            words_mask[i, :end] = 1.0
    else:
        target = None
        lengths = None
        words_mask = None
   
    return target, words_mask, idxs, cap_ids, raw_cap_ids

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
        
        elif opt.dset_name =='charades':
            self.video_feat_path = '../../depends/charades/charades_clip_i3d_numpy.hdf5'
            
        elif opt.dset_name =='Youcook2':
            self.video_feat_path = '../../depends/youcook2/video_i3d_feats.h5'
          
        self.video_feat_file = h5py.File(self.video_feat_path, 'r')
        
        self.query_feat_file = h5py.File(self.text_feat_path, 'r')

        if self.config.caption_rate > 0:
            if opt.dset_name =='charades':
                self.query_feat_file_caption_gpt = '../../depends/charades/charades_train_020_merged.hdf5'
            elif opt.dset_name =='activitynet':
                self.query_feat_file_caption_gpt = '../../depends/activitynet/anet_train_020_merged.hdf5'
            elif opt.dset_name =='Youcook2':
                self.query_feat_file_caption_gpt = '../../depends/youcook2/youcook_train_020_merged.hdf5'

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
                elif opt.dset_name == 'Youcook2':
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
            elif opt.dset_name =='Youcook2':
                self.query_feat_file_global_caption = '../../depends/youcook2/youcook_train_global.hdf5'
                
            self.query_feat_file_global_caption_from_gpt = h5py.File(self.query_feat_file_global_caption, 'r')
            all_keys = list(self.query_feat_file_global_caption_from_gpt.keys())
            self.global_caption_map = {}
            for key in all_keys:
                if opt.dset_name == 'charades':
                    video_id = key.split('_')[0]
                elif opt.dset_name == 'activitynet':
                    video_id = key.rsplit('_', 1)[0]
                elif opt.dset_name == 'Youcook2':
                    video_id = key.rsplit('_', 1)[0]
                if video_id not in self.global_caption_map:
                    self.global_caption_map[video_id] = []
                self.global_caption_map[video_id].append(key)
            print(f"[全局caption] 共 {len(all_keys)} 条，{len(self.global_caption_map)} 个视频。")
       
        # self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')


    def __getitem__(self, index):
        video_id = self.video_ids[index]
        cap_ids = self.vid_caps[video_id]

        if self.config.dset_name in {'activitynet', 'charades', 'tvr', 'Youcook2'}:
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


        return clip_video_feature, frame_video_feature, cap_tensors, index, cap_ids, video_id, global_caption_mask, local_start_end_tensor

    
    def __len__(self):
        return self.length
    
class VisDataSet4MS_SL(data.Dataset):

    def __init__(self, visual_feat, video2frames, opt, video_ids=None):
        self.visual_feat = visual_feat
        self.video2frames = video2frames
        if video_ids is not None:
            self.video_ids = video_ids
        else:
            self.video_ids = video2frames.keys()
        self.length = len(self.video_ids)
        self.map_size = opt.map_size
        self.max_ctx_len = opt.max_ctx_l
        self.config = opt
        if opt.dset_name =='activitynet':
            # self.video_feat_path = "dataset/activitynet_clip/FeatureData/i3d/feature.bin"
            # self.video_flow_feat_path = "dataset/activitynet_i3d/TextData/clip_npy_total"
            self.video_feat_path = '../../depends/activitynet/anet_clip_i3d_numpy.hdf5'

            
        elif opt.dset_name =='Youcook2':
            self.video_feat_path = '../../depends/datasets/Youcook2/video_i3d_feats.h5'
            
        elif opt.dset_name =='charades':
            self.video_feat_path = '../../depends/charades/charades_clip_i3d_numpy.hdf5'
            
        # elif opt.dset_name =='tvr':
        #     # self.video_feat_path = "dataset/tvr_i3d/FeatureData/i3d/feature.bin"
        #     # self.video_flow_feat_path = "dataset/tvr_clip/FeatureData/new_clip_vit_32_tvr_vid_features.hdf5"
        #     self.video_feat_path = '../../depends/tvr_clip_i3d_numpy.hdf5'
 
        self.video_feat_file = h5py.File(self.video_feat_path, 'r')

        # if 1 or not self.config.debug_mode:
        #     self.video_flow_feat = h5py.File(self.video_flow_feat_path, 'r')
        # self.video_feat = np.fromfile(self.video_feat_path, dtype=np.float32)
            
    def __getitem__(self, index):
        video_id = self.video_ids[index]

        if self.config.dset_name=='activitynet' or self.config.dset_name=='charades' or self.config.dset_name=='tvr' or self.config.dset_name=='Youcook2':
            temp_feat = self.video_feat_file[video_id]['i3d_feat'][...]
            clip_video_feature = average_to_fixed_length(temp_feat, self.map_size)
            clip_video_feature = torch.from_numpy(l2_normalize_np_array(clip_video_feature)).unsqueeze(0)

            frame_video_feature = uniform_feature_sampling(temp_feat, self.max_ctx_len)
            frame_video_feature = torch.from_numpy(l2_normalize_np_array(frame_video_feature))#.unsqueeze(0)

            # cap_ids = self.global_caption_map[video_id]
            # cap_tensors = []
            # for cap_id in cap_ids:
            #     cap_feat = self.global_caption_file[cap_id][...]
            #     cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_ctx_len]
            #     cap_tensors.append(cap_tensor)

            return clip_video_feature, frame_video_feature, index, video_id   # ,cap_tensor
        else:
            frame_video_feature = uniform_feature_sampling(np.load(save_path), self.max_ctx_len)
            frame_video_feature = l2_normalize_np_array(frame_video_feature)
            frame_video_feature = torch.from_numpy(frame_video_feature)

           
            
        return clip_video_feature, frame_video_feature, index, video_id
    def __len__(self):
        return self.length

class TxtDataSet4MS_SL(data.Dataset):
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

        return cap_tensor, index, cap_id, cap_id

    def __len__(self):
        return self.length

if __name__ == '__main__':
    pass




