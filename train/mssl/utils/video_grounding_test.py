import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class VideoGroundingModel(nn.Module):
    def __init__(self, dim, cross_attn_heads=4, cross_attn_layers=1):
        super().__init__()
        # Cross Attention: video为Q，text为KV
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=cross_attn_heads, batch_first=True)
        # Grounding预测头
        self.grounding_head = nn.Linear(dim, 2)  # 预测start_offset, end_offset

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

    def forward(self, text, text_mask, video, label_dict, gt_start_end):
        """
        text: bs1 * n * dim
        text_mask: bs1 * n
        video: bs2 * m * dim
        label_dict: {video_idx: [text_idx1, text_idx2, ...]}
        gt_start_end: bs1 * 2, 范围0-1
        """
        bs1, n, dim = text.shape
        bs2, m, _ = video.shape

        # 1. 扩展video到bs1
        video_exp = self.expand_video_by_label(video, label_dict, bs1)  # bs1 * m * dim

        # 2. Cross Attention: video为Q，text为KV
        # nn.MultiheadAttention: (batch, seq, dim)
        # Q: video_exp (bs1, m, dim), K/V: text (bs1, n, dim)
        # mask: key_padding_mask, 1表示被mask
        text_key_padding_mask = (text_mask == 0)  # bs1 * n, True为padding

        # MultiheadAttention输入: (batch, seq, dim)
        video_exp_t = video_exp
        text_t = text
        # attn_output: (bs1, m, dim)
        attn_output, _ = self.cross_attn(
            query=video_exp_t,
            key=text_t,
            value=text_t,
            key_padding_mask=text_key_padding_mask
        )
        video_enhanced = attn_output  # bs1 * m * dim

        # 3. Grounding预测
        grounding_pred = self.grounding_head(video_enhanced)  # bs1 * m * 2

        # 4. 计算loss
        # 对每个text，ground truth是[gt_start, gt_end]，范围0-1
        # 对每个video token，预测offset，假设token的归一化位置为pos = i / m
        pos = torch.arange(m, device=video.device).float() / m  # m,
        pos = pos.unsqueeze(0).unsqueeze(-1)  # 1 * m * 1
        pos = pos.expand(bs1, m, 1)  # bs1 * m * 1

        pred_start = pos + grounding_pred[..., 0:1]  # bs1 * m * 1
        pred_end = pos + grounding_pred[..., 1:2]    # bs1 * m * 1

        gt_start = gt_start_end[:, 0].unsqueeze(1).expand(-1, m).unsqueeze(-1)  # bs1 * m * 1
        gt_end = gt_start_end[:, 1].unsqueeze(1).expand(-1, m).unsqueeze(-1)    # bs1 * m * 1

        loss_start = F.mse_loss(pred_start, gt_start, reduction='mean')
        loss_end = F.mse_loss(pred_end, gt_end, reduction='mean')
        grounding_loss = loss_start + loss_end

        return grounding_loss

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

    model = VideoGroundingModel(input_dim)
    loss = model(text, text_mask, video, label_dict, start_end_gt)
    print("Grounding loss:", loss.item())