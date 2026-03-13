import math
import torch
import torch.nn as nn
import random

class LocalHingeLoss(nn.Module):
    def __init__(self, margin=0.2, reduction='mean'):
        self.caption_hinge_margin = margin
        self.reduction = reduction
        super(LocalHingeLoss, self).__init__()
        # TODO: cuda:0 写死了过于草率，之后再改吧
        self.start_end_map = self.precompute_window_tensors().to('cuda:0')# self.start_end_map从CPU转移到GPU

    def precompute_window_tensors(self):
        """预计算所有窗口的起始和结束位置，并存储为张量"""
        start_end_list = []
        cnt = 0
        for kernel_size in range(32):
            for start in range(32):
                if start + kernel_size > 31:
                    break
                start_end_list.append([start, start + kernel_size])
                cnt += 1
        #528个索引 0： 0，0 528 ：0,31  
        return torch.tensor(start_end_list)  #tensor的维度是 528*2
    
    def filter_caption(self, gpt_mask, raw_input_tensor, query_meta, batch_start_end_tensor):
        '''
        过滤掉query 只留下caption
        gpt_mask = torch.tensor([0,1,2,0,1,0,1,1,2])
        index_mask = gpt_mask.nonzero().squeeze(1)
        # 输出: ····tensor([0, 2, 4])
        '''
        '''
        {0: [0, 1, 2, 3, 4, 5], 1: [6, 7], 2: [8, 9, 10, 11], 3: [12, 13, 14, 15, 16], 4: [17, 18, 19, 20, 21], 5: [22, 23], 6: [24, 25, 26, 27, 28, 29, 30], 
        '''

        filtered_caption_input_tensor = []
        filtered_batch_start_end_tensor = []

        for batch_idx, idx_list in query_meta.items():
            for idx in idx_list:
                if gpt_mask[idx] == 2:
                    if batch_start_end_tensor[idx].sum() == 0:
                        raise ValueError(f"batch_start_end_tensor at idx {idx} is all zeros, which means there might be an error in the caption index")
                    filtered_caption_input_tensor.append(raw_input_tensor[idx, :, batch_idx])
                    filtered_batch_start_end_tensor.append(batch_start_end_tensor[idx])

        if len(filtered_caption_input_tensor) == 0:
            return torch.empty(0, raw_input_tensor.size(1), device=raw_input_tensor.device), torch.empty(0, 2, device=batch_start_end_tensor.device)

        filtered_caption_input_tensor = torch.stack(filtered_caption_input_tensor)
        filtered_batch_start_end_tensor = torch.stack(filtered_batch_start_end_tensor)
        
        return filtered_caption_input_tensor, filtered_batch_start_end_tensor

    def forward(self, gpt_mask, raw_input_tensor, batch_start_end_tensor, query_meta):
        
        filtered_caption_input_tensor,filtered_batch_start_end_tensor = self.filter_caption(
            gpt_mask,raw_input_tensor,query_meta,batch_start_end_tensor)
        #filtered_caption_input_tensor 维度是caption_num * 528
        #filtered_batch_start_end_tensor 维度是caption_num * 2
        #gpt mask n
        # raw input tensor  n * 528 * 128 --> n是文本数量 128是视频数量
        #batch_start_end_tensor n * 2
        #query_meta {'0':[0,1,2], '1':[3,4,5]}
        # 转换输入范围

        start_idx = (filtered_batch_start_end_tensor[:, 0] * 32).floor()
        start_idx[start_idx > 31] = 31  # 只对大于31的情况特殊处理
        filtered_batch_start_end_tensor[:, 0] = start_idx.long()
        end_idx = (filtered_batch_start_end_tensor[:, 1] * 32).ceil()
        end_idx[end_idx > 31] = 31  # 只对大于31的情况特殊处理
        filtered_batch_start_end_tensor[:, 1] = end_idx.long()

        # 检查并修正 end < start 的情况：end = max(start, end)
        start = filtered_batch_start_end_tensor[:, 0]
        end = filtered_batch_start_end_tensor[:, 1]
        filtered_batch_start_end_tensor[:, 1] = torch.max(start, end)


        # 正样本：窗口完全包含在目标范围内
        # filtered_batch_start_end_tensor 维度是caption_num * 2     self.start_end_map的维度是528*2
        # filtered_batch_start_end_tensor[:, 0].unsqueeze(1) 维度是captionnum * 1    self.start_end_map[:, 0].unsqueeze(0)  维度是1*528  -- > caption_num * 528
        # 
        pos_mask = (filtered_batch_start_end_tensor[:, 0].unsqueeze(1) == self.start_end_map[:, 0].unsqueeze(0)) \
                 & (filtered_batch_start_end_tensor[:, 1].unsqueeze(1) == self.start_end_map[:, 1].unsqueeze(0)) # [caption_num * 528]
        #pos_mask最后的维度是 caption num * 528，每一个caption只会有一个是pos mask
        no_true_indices = (pos_mask.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
        if len(no_true_indices) > 0:
            print('pos_mask全为False的行index:', no_true_indices.tolist())
        
        #neg mask 取得是和pos 区间完全没重叠的，在外面的那部分数据
        neg_mask = (filtered_batch_start_end_tensor[:, 0].unsqueeze(1) > self.start_end_map[:, 1].unsqueeze(0)) \
                 | (filtered_batch_start_end_tensor[:, 1].unsqueeze(1) < self.start_end_map[:, 0].unsqueeze(0))
                     # [caption_num * 528]
            
        # 用 float('-inf') 或 float('nan') 填充无效位置
        # 检查是否存在有效的正样本
        if not pos_mask.any():
            return torch.tensor(0.0, device=filtered_caption_input_tensor.device)
            
        # 检查是否存在有效的负样本
        if not neg_mask.any():
            return torch.tensor(0.0, device=filtered_caption_input_tensor.device)
            
        # 使用 -inf 填充无效位置
        pos_values = filtered_caption_input_tensor.masked_fill(~pos_mask, float('-inf'))
        neg_values = filtered_caption_input_tensor.masked_fill(~neg_mask, float('-inf'))

        # 直接均值
        pos_mean = pos_values.masked_fill(pos_values == float('-inf'), 0).sum(dim=1) / pos_mask.sum(dim=1).float()
        neg_mean = neg_values.masked_fill(neg_values == float('-inf'), 0).sum(dim=1) / neg_mask.sum(dim=1).float()
        pos_scores = pos_mean
        neg_scores = neg_mean

        # 计算 hinge loss
        loss = torch.clamp(neg_scores - pos_scores + self.caption_hinge_margin, min=0)
        
        # 如果 loss 中出现了 nan，返回 0
        if torch.isnan(loss).any():
            return torch.tensor(0.0, device=filtered_caption_input_tensor.device)
            
        return loss.mean()
    

    
      

"""格式如下"""
# temp = CaptionHingeLoss()
# ##query和video的每一个segment的分数
# raw_input_tensor = torch.randn(428, 528, 128) #100个是caption，328个是query
# gpt_mask = torch.randint(0, 1).repeat(428) #gpt mask  1 表示是caption，0表示是query    0,5,10是caption
# start_end = torch.randn(100, 2) #每一个caption对应的video的开始和结束时间  构造需要注意start要小于end，并且两个都在0-1之间
# query_meta = torch.randint((0, 128), size=428) #不能超过128   每一个caption(query)所对应的video的index是多少，去掉其他无关的video的信息，只保留caption对应的video的信息

if __name__ == "__main__":
    class Config:
        caption_hinge_margin = 1.0
    config = Config()
    loss_fn = LocalHingeLoss(config)

    batch_size = 128
    query_meta = {}
    gpt_mask_list = []
    idx = 0
    for batch_idx in range(batch_size):
        # 每个 batch 4-6 个 query
        n_query = random.randint(4, 6)
        indices = list(range(idx, idx + n_query))
        query_meta[batch_idx] = indices
        idx += n_query

        # 保证每个 batch 0,1,2 至少有一个
        labels = [0, 1, 2]
        if n_query > 3:
            labels += [random.randint(0, 2) for _ in range(n_query - 3)]
        random.shuffle(labels)
        gpt_mask_list.extend(labels)

    n_total = idx  # 总文本数
    gpt_mask = torch.tensor(gpt_mask_list, dtype=torch.int)

    # raw_input_tensor: n_total * 528 * batch_size
    raw_input_tensor = torch.randn(n_total, 528, batch_size).cuda()

    # 统计所有caption（gpt_mask==2）的数量
    caption_indices = [i for i, v in enumerate(gpt_mask_list) if v == 2]
    caption_num = len(caption_indices)

    # batch_start_end_tensor: caption_num * 2
    batch_start_end_tensor = torch.rand(caption_num, 2).cuda()
    batch_start_end_tensor = torch.sort(batch_start_end_tensor, dim=1)[0]

    # 构造完整的 n_total*2 的 batch_start_end_tensor，非caption部分补零
    full_batch_start_end_tensor = torch.zeros(n_total, 2).cuda()
    for i, idx in enumerate(caption_indices):
        full_batch_start_end_tensor[idx] = batch_start_end_tensor[i]

    # 前向
    loss = loss_fn(gpt_mask.cuda(), raw_input_tensor, full_batch_start_end_tensor, query_meta)
    print("Loss:", loss.item())
