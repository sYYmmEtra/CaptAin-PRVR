import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class PredictionHead(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=384, dropout=0.1, norm_type='layer'):
        super().__init__()
        self.down_proj = nn.Linear(input_dim, hidden_dim)
        
        # 可以选择 LayerNorm 或 BatchNorm
        if norm_type == 'layer':
            self.norm = nn.LayerNorm(hidden_dim)
        else:  # batch norm
            self.norm = nn.BatchNorm1d(hidden_dim)
            
        self.dropout = nn.Dropout(dropout)  # dropout 默认0.1，可调
        self.activation = nn.GELU()
        self.pred = nn.Linear(hidden_dim, 2)
        self.tanh = nn.Tanh()  # 激活函数，确保输出在 [-1,1]
    
    def forward(self, x):
        x = self.down_proj(x)
        if isinstance(self.norm, nn.BatchNorm1d):
            # BatchNorm1d 需要调整维度顺序
            x = x.transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.norm(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.pred(x)
        x = self.tanh(x)
        return x

class VideoGroundingModel(nn.Module):
    def __init__(self,l1_or_mse="none", input_dim=384, hidden_dim=384, max_len=32, cross_attn_heads=4, dropout=0.1, norm_type='layer'):
        super().__init__()
        self.max_len = max_len
        self.l1_or_mse = l1_or_mse
        # 可学习的位置编码
        self.pos_embed = nn.Embedding(max_len, input_dim)
        
        # 多头注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=cross_attn_heads,
            batch_first=True,
            dropout=dropout  # 给 attention 也加上 dropout
        )
        
        # 预测头
        self.grounding_head = PredictionHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            norm_type=norm_type
        )

    def expand_video_by_label(self, video, label_dict, bs1):
        """
        根据label_dict扩展video到bs1
        """
        device = video.device
        bs2, m, dim = video.shape
        video_expanded = []
        # label_dict: {video_idx: [text_idx1, text_idx2, ...]}
        # 先构建text_idx到video_idx的映射
        text_to_video = [None] * bs1
        for vidx, tlist in label_dict.items():
            for tidx in tlist:
                text_to_video[tidx] = vidx
        for tidx in range(bs1):
            vidx = text_to_video[tidx]
            video_expanded.append(video[vidx])
        video_expanded = torch.stack(video_expanded, dim=0)  # bs1 * m * dim
        return video_expanded

    def filter_video_by_global_caption_mask(self, text, text_mask, label_dict, gt_start_end, global_caption_mask):
        # 1. 找到所有caption的索引（local_caption_mask==2）
        caption_indices = [i for i, v in enumerate(global_caption_mask) if v == 2]
        if len(caption_indices) == 0:
            # 没有caption，返回空
            return text[:0], text_mask[:0], {}, gt_start_end[:0]
        # 2. 筛选text、text_mask、gt_start_end
        filtered_text = text[caption_indices]
        filtered_text_mask = text_mask[caption_indices]
        filtered_gt_start_end = gt_start_end[caption_indices]
        # 3. 更新label_dict
        # label_dict: {video_idx: [text_idx1, text_idx2, ...]}
        # 只保留caption_indices的映射
        filtered_label_dict = {}
        idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(caption_indices)}
        for vid, idx_list in label_dict.items():
            new_list = [idx_map[idx] for idx in idx_list if idx in idx_map]
            if new_list:
                filtered_label_dict[vid] = new_list
        return filtered_text, filtered_text_mask, filtered_label_dict, filtered_gt_start_end

    def forward(self, text, text_mask, video, label_dict, gt_start_end, global_caption_mask):
        text, text_mask, label_dict, gt_start_end = self.filter_video_by_global_caption_mask(
            text, text_mask, label_dict, gt_start_end, global_caption_mask)
        """
        text: bs1 * n * dim
        text_mask: bs1 * n
        video: [128, 32, 384]
        label_dict: {video_idx: [text_idx1, text_idx2, ...]}
        gt_start_end: bs1 * 2, 范围0-1
        """
         
        bs1, n, dim = text.shape
        bs2, m, _ = video.shape
            
        # 1. 扩展video到bs1
        video_exp = self.expand_video_by_label(video, label_dict, bs1)  # bs1 * m * dim

        
        # 2. 添加位置编码
        position_ids = torch.arange(m, device=video.device)
        pos_embed = self.pos_embed(position_ids)  # [m, dim]
        pos_embed = pos_embed.unsqueeze(0).expand(bs1, -1, -1)  # [bs1, m, dim]
        video_exp = video_exp + pos_embed  # 加入位置信息

        # 3. Cross Attention
        text_key_padding_mask = (text_mask == 0)
        attn_output, _ = self.cross_attn(
            query=video_exp,
            key=text,
            value=text,
            key_padding_mask=text_key_padding_mask
        )
        
        # 4. 预测和loss计算
        grounding_pred = self.grounding_head(attn_output)  # [bs1, m, 2]
        
        gt = gt_start_end.unsqueeze(2).expand(-1, -1, m)  # [bs1, 2, m]
        pos = (torch.arange(m, device=video.device).float()) / (m-1)  # 加上0.5表示取中心点
        pos = pos.unsqueeze(0).unsqueeze(0)  # [1, 1, m]
        pos = pos.expand(bs1, 2, m)  # [bs1, 2, m]

        gt_offset = gt - pos  # [bs1, 2, m]
        # import pdb;pdb.set_trace()
        if self.l1_or_mse == "l1":
            loss = F.smooth_l1_loss(grounding_pred.transpose(-1, -2), gt_offset, reduction='mean')
        elif self.l1_or_mse == "mse":
            loss = F.mse_loss(grounding_pred.transpose(-1, -2), gt_offset, reduction='mean')
        else :
            print("invalid l1_or_mse")
            exit()


        return loss

# 用法示例
if __name__ == "__main__":
    # 参数设置
    input_dim = 32
    batch_size = 3
    caption_num = 5
    clip_len = 8
    max_len = 10

    # 视频特征 [3, 8, 32]
    video = torch.randn(batch_size, clip_len, input_dim)

    # 文本特征与mask [5, 10, 32]
    text = torch.randn(caption_num, max_len, input_dim)
    text_mask = torch.ones(caption_num, max_len, dtype=torch.bool)
    for i in range(caption_num):
        real_len = random.randint(6, max_len)
        text_mask[i, real_len:] = 0  # 将 padding 部分置为 0

    # label_dict：每个视频对应哪些 captions
    label_dict = {
        0: [0, 1],
        1: [2, 3],
        2: [4]
    }

    # start_end_gt：每条 caption 的 [start, end]（归一化时间位置）
    start_end_gt = torch.zeros(caption_num, 2)
    for i in range(caption_num):
        start = random.uniform(0.0, 0.6)
        end = random.uniform(start + 0.1, 1.0)
        start_end_gt[i] = torch.tensor([start, end])

    # 添加 global_caption_mask
    # 2 表示是 caption，0 表示不是
    global_caption_mask = torch.tensor([2, 2, 2, 0, 0])  # 假设前3个是caption

    model = VideoGroundingModel(input_dim)
    loss = model(text, text_mask, video, label_dict, start_end_gt, global_caption_mask)
    print("Grounding loss:", loss.item())