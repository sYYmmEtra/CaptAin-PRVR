import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb
from easydict import EasyDict as edict
from method.model_components import BertAttention, LinearLayer, BertSelfAttention, TrainablePositionalEncoding
from method.model_components import clip_nce, frame_nce
from method.model_loss import LocalHingeLoss
from method.video_grounding_final import VideoGroundingModel


class MS_SL_Net(nn.Module):
    def __init__(self, config):
        super(MS_SL_Net, self).__init__()
        self.config = config

        self.query_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_desc_l,
                                                           hidden_size=config.hidden_size, dropout=config.input_drop)
        self.clip_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                         hidden_size=config.hidden_size, dropout=config.input_drop)
        self.frame_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                          hidden_size=config.hidden_size, dropout=config.input_drop)
        #
        self.query_input_proj = LinearLayer(config.query_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        #
        self.query_encoder = BertAttention(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop))


        self.clip_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.clip_encoder = copy.deepcopy(self.query_encoder)

        self.frame_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                             dropout=config.input_drop, relu=True)
        self.frame_encoder = copy.deepcopy(self.query_encoder)

        self.modular_vector_mapping = nn.Linear(config.hidden_size, out_features=1, bias=False)


        self.pool_layers = nn.ModuleList([nn.Identity()]
                                         + [nn.AvgPool1d(i, stride=1) for i in range(2, config.map_size + 1)]
                                         )

        self.mapping_linear = nn.ModuleList([nn.Linear(config.hidden_size, out_features=config.hidden_size, bias=False)
                                             for i in range(2)])

        self.clip_nce_criterion = clip_nce(reduction='mean')
        self.video_nce_criterion = frame_nce(reduction='mean')

        self.local_hinge_loss = LocalHingeLoss(margin=self.config.local_margin, reduction='mean')
        # self.video_grounding_loss = VideoGroundingModel(l1_or_mse=self.config.l1_or_mse)
        self.reset_parameters()

    def reset_parameters(self):
        """ Initialize the weights."""

        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(re_init)

    def set_hard_negative(self, use_hard_negative, hard_pool_size):
        """use_hard_negative: bool; hard_pool_size: int, """
        self.config.use_hard_negative = use_hard_negative
        self.config.hard_pool_size = hard_pool_size



    def forward(self, clip_video_feat, frame_video_feat, frame_video_mask, query_feat, query_mask, query_labels,global_caption_mask,local_start_end_tensor,current_epoch):
        
        encoded_frame_feat, vid_proposal_feat, encoded_clip_feat = self.encode_context(
            clip_video_feat, frame_video_feat, frame_video_mask)
        
        clip_scale_scores, frame_scale_scores, clip_scale_scores_, frame_scale_scores_,max_clip_scale_context_scores,max_clip_scale_context_scores_ ,query_labels_,global_clip_scores,raw_clip_scale_scores,encoded_query\
            = self.get_pred_from_raw_query(
            query_feat, query_mask, query_labels, vid_proposal_feat, encoded_frame_feat, frame_video_mask, return_query_feats=True, global_caption_mask=global_caption_mask)
        
        raw_label_dict = {}
        for index, label in enumerate(query_labels):
            if label in raw_label_dict:
                raw_label_dict[label].append(index)
            else:
                raw_label_dict[label] = []
                raw_label_dict[label].append(index)
        
        # 根据当前epoch决定是否使用其他loss
        if current_epoch < self.config.pretrain_grounding_epochs and self.config.video_grounding_weight > 0.0:
            print(f"current_epoch = {current_epoch}")
            # 预训练阶段,只使用video_grounding_loss
            video_grounding_loss = self.config.video_grounding_weight * self.video_grounding_loss(encoded_query,query_mask,encoded_clip_feat,raw_label_dict,local_start_end_tensor,global_caption_mask)
            loss = video_grounding_loss
            return loss, {"loss_overall": float(loss), 
                         "video_grounding_loss": video_grounding_loss,
                         "clip_nce_loss": 0.0,
                         "clip_trip_loss": 0.0, 
                         "frame_nce_loss": 0.0,
                         "frame_trip_loss": 0.0,
                         "local_hinge_loss": 0.0,
                         "global_hinge_loss": 0.0,
                         "global_nce_loss": 0.0,
                         "global_soft_pos_loss": 0.0,
                         "global_soft_neg_loss": 0.0}
        
        # 正常训练阶段,使用所有loss
        label_dict = {}
        for index, label in enumerate(query_labels_):
            if label in label_dict:
                label_dict[label].append(index)
            else:
                label_dict[label] = []
                label_dict[label].append(index)

        clip_nce_loss = 0.02 * self.clip_nce_criterion(query_labels_, label_dict, clip_scale_scores_)
        clip_trip_loss = self.get_clip_triplet_loss(clip_scale_scores, query_labels_)


        frame_nce_loss = 0.04 * self.video_nce_criterion(frame_scale_scores_)
        frame_trip_loss = self.get_frame_trip_loss(frame_scale_scores)
        
        
        if self.config.video_grounding_weight > 0.0:
            video_grounding_loss = self.config.video_grounding_weight * self.video_grounding_loss(encoded_query,query_mask,encoded_clip_feat,raw_label_dict,local_start_end_tensor,global_caption_mask)
        else:
            video_grounding_loss = 0

        if self.config.local_hinge_weight > 0.0:
            local_hinge_loss = self.config.local_hinge_weight * self.local_hinge_loss(global_caption_mask,raw_clip_scale_scores,local_start_end_tensor,raw_label_dict)
        else:
            local_hinge_loss = 0
            
        if self.config.global_nce_weight > 0.0:        
            global_nce_loss = self.config.global_nce_weight * self.global_caption_infonce_loss(max_clip_scale_context_scores_, query_labels, label_dict, global_caption_mask)
        else:
            global_nce_loss = 0
            
        if self.config.global_hinge_weight > 0.0:
            global_hinge_loss = self.config.global_hinge_weight * self.global_caption_hinge_loss(max_clip_scale_context_scores, query_labels, global_caption_mask)
        else:
            global_hinge_loss = 0

        if self.config.global_soft_pos_weight > 0.0 or self.config.global_soft_neg_weight > 0.0:
            pos_loss, neg_loss = self.global_soft_loss(global_clip_scores, query_labels, global_caption_mask)
            global_soft_pos_loss = self.config.global_soft_pos_weight * pos_loss
            global_soft_neg_loss = self.config.global_soft_neg_weight * neg_loss
        else:
            global_soft_pos_loss = 0
            global_soft_neg_loss = 0

        if self.config.grounding_loss_only_in_pretrain:
            loss =  clip_nce_loss + clip_trip_loss + frame_nce_loss + frame_trip_loss + \
                    local_hinge_loss + \
                    global_hinge_loss + global_nce_loss + global_soft_pos_loss + global_soft_neg_loss
        else :            
            loss =  clip_nce_loss + clip_trip_loss + frame_nce_loss + frame_trip_loss + \
                    local_hinge_loss + video_grounding_loss + \
                    global_hinge_loss + global_nce_loss + global_soft_pos_loss + global_soft_neg_loss

        return loss, {"loss_overall": float(loss), 'clip_nce_loss': clip_nce_loss,
                      'clip_trip_loss': clip_trip_loss,
                      'frame_nce_loss': frame_nce_loss, 'frame_trip_loss': frame_trip_loss,
                      'local_hinge_loss': local_hinge_loss,
                      'video_grounding_loss':video_grounding_loss,
                      'global_hinge_loss': global_hinge_loss,
                      'global_nce_loss': global_nce_loss,
                      'global_soft_pos_loss': global_soft_pos_loss,
                      'global_soft_neg_loss': global_soft_neg_loss}



    def encode_query(self, query_feat, query_mask):
        # import ipdb;ipdb.set_trace()
        encoded_query = self.encode_input(query_feat, query_mask, self.query_input_proj, self.query_encoder,
                                          self.query_pos_embed)  # (N, Lq, D) [query_num, desc_len, 384]
        video_query = self.get_modularized_queries(encoded_query, query_mask)  # (N, D) * 1

        return video_query,encoded_query

    def encode_context(self, clip_video_feat, frame_video_feat, video_mask=None):


        encoded_clip_feat = self.encode_input(clip_video_feat, None, self.clip_input_proj, self.clip_encoder,
                                               self.clip_pos_embed)

        encoded_frame_feat = self.encode_input(frame_video_feat, video_mask, self.frame_input_proj,
                                                self.frame_encoder,
                                                self.frame_pos_embed)

        vid_proposal_feat_map = self.encode_feat_map(encoded_clip_feat)


        return encoded_frame_feat, vid_proposal_feat_map,encoded_clip_feat # [128, 32, 384]


    def encode_feat_map(self, x_feat):
        batch_size, seq_len, feat_dim = x_feat.shape
        pool_in = x_feat.permute(0, 2, 1)

        proposal_feat_map = []
        # index_ranges = []
        for idx, pool in enumerate(self.pool_layers):
            x = pool(pool_in).permute(0, 2, 1)
            proposal_feat_map.append(x)
            # 计算当前池化操作对应的原始索引范围
            # if idx == 0:  # nn.Identity() 直接对应原始索引
            #     ranges = [(i, i) for i in range(seq_len)]
            # else:
            #     kernel_size = idx + 1  # 从2开始的池化窗口大小
            #     ranges = [(i, i + kernel_size - 1) for i in range(seq_len - kernel_size + 1)]
            
            # index_ranges.append(ranges)
        proposal_feat_map = torch.cat(proposal_feat_map, dim=1)


        return proposal_feat_map


    @staticmethod
    def encode_input(feat, mask, input_proj_layer, encoder_layer, pos_embed_layer):
        """
        Args:
            feat: (N, L, D_input), torch.float32
            mask: (N, L), torch.float32, with 1 indicates valid query, 0 indicates mask
            input_proj_layer: down project input
            encoder_layer: encoder layer
            pos_embed_layer: positional embedding layer
        """
        feat = input_proj_layer(feat)
        feat = pos_embed_layer(feat)

        if mask is not None:
            mask = mask.unsqueeze(1)  # (N, 1, L), torch.FloatTensor
        return encoder_layer(feat, mask)  # (N, L, D_hidden)

    def get_modularized_queries(self, encoded_query, query_mask):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        modular_attention_scores = self.modular_vector_mapping(encoded_query)  # (N, L, 2 or 1)
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)  # (N, 2 or 1, D)
        return modular_queries.squeeze()


    @staticmethod
    def get_clip_scale_scores(modularied_query, context_feat): # [query,384] [video,528,384]
        modularied_query = F.normalize(modularied_query, dim=-1)
        context_feat = F.normalize(context_feat, dim=-1)

        clip_level_query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0) # [query,528,video]
        max_clip_scale_context_scores = clip_level_query_context_scores[:, -1, :]  # shape: [query,video] 取第 528 个 clip
        
        query_context_scores, indices = torch.max(clip_level_query_context_scores, dim=1)  # (N, N) diagonal positions are positive pairs
        return query_context_scores, indices, max_clip_scale_context_scores, clip_level_query_context_scores


    def get_train_text_max_segment(self,query_feat,query_mask, context_feat_3d,topk_seg,labels): # [query,384] [video,528,384]

        
        video_query,_ = self.encode_query(query_feat, query_mask)
        video_query = F.normalize(video_query, dim=-1)
        context_feat = F.normalize(context_feat_3d, dim=-1).reshape(-1,context_feat_3d.shape[-1]) # [video*528,384]
        clip_level_query_context_scores = torch.matmul(video_query, context_feat.t()) # [query,528*video]
        """
        目前是在这里点乘计算相似度，要换成用索引找query对应的video，返回其中topk个clip（是segment不是相似度得分）
        labels = [0, 0, 1, 1, 2, 2, 3...] 表示第0，1个caption是第0号视频的 已经截断 仅仅包含这一轮的caption
        label_dict: {video_idx: [text_idx1, text_idx2, ...]} 全部的video和全部的caption
        batch_start_index 这轮第一个caption在全部caption中的index
        clip_level_query_context_scores [query,video*528] -> [query,video,528] --(label_dict)--> [query,528] --(topk)--> [query,topk_seg]
        每次调用这个函数处理一个batch的caption，我希望能找到这轮caption对应的video。
        """
        
        # 1.reshape [query,video*528] -> [query,video,528]
        video_num,clip_num,dim = context_feat_3d.shape
        num_video = context_feat.shape[0] // clip_num # 计算 video 数量
        clip_level_query_context_scores_3d = clip_level_query_context_scores.view(-1, num_video, 528)  # [query, video, 528]
        # import ipdb;ipdb.set_trace()
        # 2.找出每个 query 对应的是哪个 video
        labels_tensor = torch.tensor(labels, device=clip_level_query_context_scores_3d.device)  # [query] [0, 0, 1, 1, 2, 2, 3...]
        labels_tensor_2d = labels_tensor.unsqueeze(1) * clip_num
        query_indices = torch.arange(len(labels), device=clip_level_query_context_scores_3d.device)  # [query]

        # 提取每个 query 对应 video 的得分 [query, 528]
        clip_scores_per_query = clip_level_query_context_scores_3d[query_indices, labels_tensor]  # [query, 528]
        
        _, indices_topk = torch.topk(clip_scores_per_query, k=topk_seg, dim=1)  # shape: [query, k]
        indices_topk += labels_tensor_2d
        # query_context_scores_max = context_feat[indices_max] # shape: [query, 384]
        query_context_scores_topk = context_feat[indices_topk] # shape: [query, k, 384]

        
        # return video_query,query_context_scores_max, query_context_scores_topk # 3个都是归一化的
        return video_query, query_context_scores_topk # pooling降维后的query. 对于query topk个 clip tensors （2个都是归一化的）

    @staticmethod
    def get_global2query_scores(video_query, global_caption_mask):
        """
        计算 global caption（video_query[global_caption_mask==1]）和所有 video_query 的相似度分数
        Args:
            video_query: (N, D) 所有 query 的特征
            global_caption_mask: (N,) 0/1/2，1表示global caption
        Returns:
            global_caption_scores: (num_global, N) 每个global caption和所有video_query的相似度分数
            global_indices: (num_global,) global caption在video_query中的索引
        """
        # 选出global caption部分
        device = video_query.device
        global_caption_mask_tensor = torch.tensor(global_caption_mask, device=device)
        global_indices = (global_caption_mask_tensor == 1).nonzero(as_tuple=True)[0]
        global_query = video_query[global_indices]  # (num_global, D)

        # 归一化
        global_query = F.normalize(global_query, dim=-1)
        video_query_norm = F.normalize(video_query, dim=-1)

        # 计算相似度 (num_global, N)
        global_caption_scores = torch.matmul(global_query, video_query_norm.t())

        return global_caption_scores, global_indices

    @staticmethod
    def get_unnormalized_clip_scale_scores(modularied_query, context_feat):

        query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)
        
        max_clip_scale_context_scores = query_context_scores[:, -1, :]

        query_context_scores, _ = torch.max(query_context_scores, dim=1)

        return query_context_scores, max_clip_scale_context_scores

    def key_clip_guided_attention(self, frame_feat, proposal_feat, feat_mask, max_index, query_labels):
        selected_max_index = max_index[[i for i in range(max_index.shape[0])], query_labels]

        expand_frame_feat = frame_feat[query_labels]

        expand_proposal_feat = proposal_feat[query_labels]

        key = self.mapping_linear[0](expand_frame_feat)
        query = expand_proposal_feat[[i for i in range(key.shape[0])], selected_max_index, :].unsqueeze(-1)
        value = self.mapping_linear[1](expand_frame_feat)

        if feat_mask is not None:
            expand_feat_mask = feat_mask[query_labels]
            scores = torch.bmm(key, query).squeeze()
            masked_scores = scores.masked_fill(expand_feat_mask.eq(0), -1e9).unsqueeze(1)
            masked_scores = nn.Softmax(dim=-1)(masked_scores)
            attention_feat = torch.bmm(masked_scores, value).squeeze()
        else:
            scores = nn.Softmax(dim=-1)(torch.bmm(key, query).transpose(1, 2))
            attention_feat = torch.bmm(scores, value).squeeze()

        return attention_feat

    def key_clip_guided_attention_in_inference(self, frame_feat, proposal_feat, feat_mask, max_index):
        key = self.mapping_linear[0](frame_feat)
        value = self.mapping_linear[1](frame_feat)
        num_vid = frame_feat.shape[0]

        index = torch.arange(num_vid).unsqueeze(1)
        query = proposal_feat[index, max_index.t()]
        if feat_mask is not None:
            scores = torch.bmm(key, query.transpose(2, 1))
            masked_scores = scores.masked_fill(feat_mask.unsqueeze(-1).eq(0), -1e9)
            masked_scores = nn.Softmax(dim=1)(masked_scores)
            attention_feat = torch.bmm(masked_scores.transpose(1, 2), value)
        else:
            scores = torch.bmm(key, query.transpose(2, 1))
            scores = nn.Softmax(dim=1)(scores)
            attention_feat = torch.bmm(scores.transpose(1, 2), value)

        return attention_feat



    def get_pred_from_raw_query(self, query_feat, query_mask, query_labels=None,
                                video_proposal_feat=None,
                                video_feat=None,
                                video_feat_mask=None,
                                return_query_feats=False,
                                global_caption_mask=None):

        
        video_query,encoded_query = self.encode_query(query_feat, query_mask)
        avg_sim = torch.nn.functional.cosine_similarity(video_query.unsqueeze(1), video_query.unsqueeze(0), dim=-1).masked_fill(torch.eye(video_query.shape[0], device=video_query.device, dtype=torch.bool), 0).sum() / (video_query.shape[0] * (video_query.shape[0] - 1))
        # import ipdb;ipdb.set_trace()

        # get clip-level retrieval scores
        # 新增了max_clip_scale_context_scores，用于计算global caption的loss
        clip_scale_scores, key_clip_indices, max_clip_scale_context_scores, clip_level_query_context_scores = \
            self.get_clip_scale_scores( video_query, video_proposal_feat)
            

        if return_query_feats:
            # 新增了max_clip_scale_context_scores，用于计算global caption的loss
            clip_scale_scores_, max_clip_scale_context_scores_ = self.get_unnormalized_clip_scale_scores(video_query, video_proposal_feat)
            
            raw_clip_scale_scores = clip_level_query_context_scores
            
            if global_caption_mask is not None:
                global_caption_mask_tensor = torch.tensor(global_caption_mask, device=video_query.device)
                global_indices = (global_caption_mask_tensor == 1).nonzero(as_tuple=True)[0]
                global_clip_scores = clip_level_query_context_scores[global_indices]
                if(self.config.query_or_caption == 0):
                    keep_indices = (global_caption_mask_tensor == 0).nonzero(as_tuple=True)[0] # 0是query 1是global 2是local 3是all
                elif(self.config.query_or_caption == 1):
                    keep_indices = (global_caption_mask_tensor != 1).nonzero(as_tuple=True)[0]
                elif(self.config.query_or_caption == 2):
                    keep_indices = (global_caption_mask_tensor != 2).nonzero(as_tuple=True)[0]
                elif(self.config.query_or_caption == 3):
                    keep_indices = (global_caption_mask_tensor != 3).nonzero(as_tuple=True)[0]
                else:
                    print("query_or_caption 参数无效")
                    exit()

                video_query = video_query[keep_indices]
                key_clip_indices = key_clip_indices[keep_indices]
                clip_scale_scores = clip_scale_scores[keep_indices]
                clip_scale_scores_ = clip_scale_scores_[keep_indices]
                query_labels_ = [query_labels[i] for i in keep_indices.tolist()]  
                   
            frame_scale_feat = self.key_clip_guided_attention(video_feat, video_proposal_feat, video_feat_mask,
                                                          key_clip_indices, query_labels_)
            frame_scale_scores = torch.matmul(F.normalize(video_query, dim=-1),
                                              F.normalize(frame_scale_feat, dim=-1).t())
            
            frame_scale_scores_ = torch.matmul(video_query, frame_scale_feat.t())

            return clip_scale_scores, frame_scale_scores, clip_scale_scores_,frame_scale_scores_,max_clip_scale_context_scores,max_clip_scale_context_scores_,query_labels_,global_clip_scores,raw_clip_scale_scores,encoded_query
        else:
            frame_scale_feat = self.key_clip_guided_attention_in_inference(video_feat, video_proposal_feat, video_feat_mask,
                                                                       key_clip_indices).to(video_query.device)
            frame_scales_cores_ = torch.mul(F.normalize(frame_scale_feat, dim=-1),
                                            F.normalize(video_query, dim=-1).unsqueeze(0))
            frame_scale_scores = torch.sum(frame_scales_cores_, dim=-1).transpose(1, 0)

            return clip_scale_scores, frame_scale_scores, key_clip_indices

    def get_clip_triplet_loss(self, query_context_scores, labels):
        v2t_scores = query_context_scores.t()
        t2v_scores = query_context_scores
        labels = np.array(labels)

        # cal_v2t_loss
        v2t_loss = 0
        for i in range(v2t_scores.shape[0]):
            pos_pair_scores = torch.mean(v2t_scores[i][np.where(labels == i)])


            neg_pair_scores, _ = torch.sort(v2t_scores[i][np.where(labels != i)[0]], descending=True)
            if self.config.use_hard_negative:
                sample_neg_pair_scores = neg_pair_scores[0]
            else:
                v2t_sample_max_idx = neg_pair_scores.shape[0]
                sample_neg_pair_scores = neg_pair_scores[
                    torch.randint(0, v2t_sample_max_idx, size=(1,)).to(v2t_scores.device)]



            v2t_loss += (self.config.margin + sample_neg_pair_scores - pos_pair_scores).clamp(min=0).sum()

        # cal_t2v_loss
        text_indices = torch.arange(t2v_scores.shape[0]).to(t2v_scores.device)
        t2v_pos_scores = t2v_scores[text_indices, labels]
        mask_score = copy.deepcopy(t2v_scores.data)
        mask_score[text_indices, labels] = 999
        _, sorted_scores_indices = torch.sort(mask_score, descending=True, dim=1)
        t2v_sample_max_idx = min(1 + self.config.hard_pool_size,
                                 t2v_scores.shape[1]) if self.config.use_hard_negative else t2v_scores.shape[1]
        sample_indices = sorted_scores_indices[
            text_indices, torch.randint(1, t2v_sample_max_idx, size=(t2v_scores.shape[0],)).to(t2v_scores.device)]

        t2v_neg_scores = t2v_scores[text_indices, sample_indices]

        t2v_loss = (self.config.margin + t2v_neg_scores - t2v_pos_scores).clamp(min=0)

        return t2v_loss.sum() / len(t2v_scores) + v2t_loss / len(v2t_scores)

    def get_frame_trip_loss(self, query_context_scores):
        """ ranking loss between (pos. query + pos. video) and (pos. query + neg. video) or (neg. query + pos. video)
        Args:
            query_context_scores: (N, N), cosine similarity [-1, 1],
                Each row contains the scores between the query to each of the videos inside the batch.
        """

        bsz = len(query_context_scores)

        diagonal_indices = torch.arange(bsz).to(query_context_scores.device)
        pos_scores = query_context_scores[diagonal_indices, diagonal_indices]  # (N, )
        query_context_scores_masked = copy.deepcopy(query_context_scores.data)
        # impossibly large for cosine similarity, the copy is created as modifying the original will cause error
        query_context_scores_masked[diagonal_indices, diagonal_indices] = 999
        pos_query_neg_context_scores = self.get_neg_scores(query_context_scores, query_context_scores_masked)
        neg_query_pos_context_scores = self.get_neg_scores(query_context_scores.transpose(0, 1),
                                                           query_context_scores_masked.transpose(0, 1))
        loss_neg_ctx = self.get_ranking_loss(pos_scores, pos_query_neg_context_scores)
        loss_neg_q = self.get_ranking_loss(pos_scores, neg_query_pos_context_scores)
        return loss_neg_ctx + loss_neg_q

    def get_neg_scores(self, scores, scores_masked):
        """
        scores: (N, N), cosine similarity [-1, 1],
            Each row are scores: query --> all videos. Transposed version: video --> all queries.
        scores_masked: (N, N) the same as scores, except that the diagonal (positive) positions
            are masked with a large value.
        """

        bsz = len(scores)
        batch_indices = torch.arange(bsz).to(scores.device)

        _, sorted_scores_indices = torch.sort(scores_masked, descending=True, dim=1)

        sample_min_idx = 1  # skip the masked positive

        sample_max_idx = min(sample_min_idx + self.config.hard_pool_size, bsz) if self.config.use_hard_negative else bsz

        # sample_max_idx = 2

        # (N, )
        sampled_neg_score_indices = sorted_scores_indices[batch_indices, torch.randint(sample_min_idx, sample_max_idx,
                                                                                       size=(bsz,)).to(scores.device)]

        sampled_neg_scores = scores[batch_indices, sampled_neg_score_indices]  # (N, )
        return sampled_neg_scores

    def get_ranking_loss(self, pos_score, neg_score):
        """ Note here we encourage positive scores to be larger than negative scores.
        Args:
            pos_score: (N, ), torch.float32
            neg_score: (N, ), torch.float32
        """
        return torch.clamp(self.config.margin + neg_score - pos_score, min=0).sum() / len(pos_score)

    def global_caption_hinge_loss(self, max_clip_scale_context_scores, query_labels, global_caption_mask):
        device = max_clip_scale_context_scores.device
        global_caption_mask = torch.tensor(global_caption_mask, device=device)
        mask_indices = (global_caption_mask == 1).nonzero(as_tuple=True)[0]
        if len(mask_indices) == 0:
            return torch.tensor(0.0, device=device)
        scores = max_clip_scale_context_scores[mask_indices]  # [num_global, num_video]
        labels = torch.tensor(query_labels, device=device)[mask_indices]  # [num_global]

        # t2v loss（原有实现）
        t2v_loss = 0.0
        for i in range(scores.shape[0]):
            pos_score = scores[i, labels[i]]
            neg_scores = torch.cat([scores[i, :labels[i]], scores[i, labels[i]+1:]])
            if self.config.use_hard_negative and neg_scores.numel() > 0:
                neg_score = neg_scores.max()
            elif neg_scores.numel() > 0:
                neg_score = neg_scores[torch.randint(0, neg_scores.numel(), (1,)).item()]
            else:
                continue
            t2v_loss += torch.clamp(self.config.global_margin + neg_score - pos_score, min=0)
        t2v_loss = t2v_loss / scores.shape[0]

        # v2t loss（新增）
        v2t_loss = 0.0
        num_videos = scores.shape[1]
        for vid_idx in range(num_videos):
            # 找到所有正样本（labels==vid_idx）的行
            pos_indices = (labels == vid_idx).nonzero(as_tuple=True)[0]
            if len(pos_indices) == 0:
                continue
            pos_scores = scores[pos_indices, vid_idx]
            # 负样本：labels!=vid_idx
            neg_indices = (labels != vid_idx).nonzero(as_tuple=True)[0]
            if len(neg_indices) == 0:
                continue
            neg_scores = scores[neg_indices, vid_idx]
            if self.config.use_hard_negative:
                neg_score = neg_scores.max()
            else:
                neg_score = neg_scores[torch.randint(0, neg_scores.numel(), (1,)).item()]
            pos_score = pos_scores.mean()  # 取均值
            v2t_loss += torch.clamp(self.config.global_margin + neg_score - pos_score, min=0)
        v2t_loss = v2t_loss / scores.shape[1]

        return t2v_loss + v2t_loss

    def global_caption_infonce_loss(self, max_clip_scale_context_scores_, query_labels, label_dict, global_caption_mask):
        device = max_clip_scale_context_scores_.device
        global_caption_mask = torch.tensor(global_caption_mask, device=device)
        mask_indices = (global_caption_mask == 1).nonzero(as_tuple=True)[0]
        if len(mask_indices) == 0:
            return torch.tensor(0.0, device=device)
        logits = max_clip_scale_context_scores_[mask_indices]  # [num_global, num_video]
        labels = [query_labels[i] for i in mask_indices.tolist()]

        # 重新构建 label_dict_new
        label_dict_new = {}
        for idx, label in enumerate(labels):
            if label not in label_dict_new:
                label_dict_new[label] = []
            label_dict_new[label].append(idx)

        return self.clip_nce_criterion(labels, label_dict_new, logits)

    def global_soft_loss(self, global_clip_scores, query_labels, global_caption_mask):
        device = global_clip_scores.device
        global_caption_mask_tensor = torch.tensor(global_caption_mask, device=device)
        # Find indices of global queries in the ORIGINAL query_labels/global_caption_mask
        global_indices_original = (global_caption_mask_tensor == 1).nonzero(as_tuple=True)[0]

        if len(global_indices_original) == 0:
            return torch.tensor(0.0, device=device)

        # global_clip_scores is already filtered in forward, so its first dimension corresponds to global queries
        # Extract labels corresponding to global queries from the ORIGINAL query_labels
        global_labels = torch.tensor(query_labels, device=device)[global_indices_original] # [num_global]

        global_scores = global_clip_scores # [num_global, 528, num_videos]

        num_global_queries = global_scores.shape[0]
        num_videos = global_scores.shape[2]
        total_pos_loss = torch.tensor(0.0, device=device)
        total_neg_loss = torch.tensor(0.0, device=device)

        for i in range(num_global_queries):
            # The i-th row of global_scores corresponds to the global query with label global_labels[i]
            target_video_idx = global_labels[i]
            
            # Positive score (528th element for the target video)
            pos_score = global_scores[i, 527, target_video_idx] # scalar

            # Weak scores (first 527 elements for the target video)
            weak_scores = global_scores[i, :150, target_video_idx] # [150] max scale clip = 5 clip

            # Negative scores (all scores not for the target video)
            neg_scores_all = torch.cat([global_scores[i, :, :target_video_idx].flatten(),
                                        global_scores[i, :, target_video_idx+1:].flatten()]) # flatten all scores from other videos 127x528=67056

            if neg_scores_all.numel() == 0:
                # Should not happen if num_videos > 1, but handle edge case
                continue
            
            # Randomly sample one negative score for this query
            neg_score = torch.mean(neg_scores_all) # scalar

            # Calculate loss for each weak score using double hinge loss (vectorized)
            # Hinge loss 1: Encourage pos_score > weak_score + margin
            loss1_batch = torch.clamp(self.config.soft_pos_margin + weak_scores - pos_score, min=0)
            # Hinge loss 2: Encourage weak_score > neg_score + margin
            loss2_batch = torch.clamp(self.config.soft_neg_margin + neg_score - weak_scores, min=0)
            
            # Sum the losses for this query's weak scores
            total_pos_loss += torch.sum(loss1_batch)
            total_neg_loss += torch.sum(loss2_batch)

        # Average loss over all global queries (and all weak scores within)
        # Total number of weak scores across all global queries is num_global_queries * 527
        # We add the sum of losses for each weak score to total_loss, so we should divide by the total number of weak samples.
        num_weak_samples = num_global_queries * weak_scores.numel() if num_global_queries > 0 and weak_scores.numel() > 0 else 1  #128x527=67456
        pos_loss = total_pos_loss / num_weak_samples
        neg_loss = total_neg_loss / num_weak_samples
        return pos_loss, neg_loss
    
    def get_query_global_cap_scores(self,query,query_mask,global_caption_feat,labels,query_metas_raw,video_metas,local_weight):
        query_feat, _ = self.encode_query(query,query_mask)
        # 归一化
        query_feat_norm = F.normalize(query_feat, dim=-1) # [query_n, 384]
        global_caption_feat_norm = F.normalize(global_caption_feat, dim=-1) # [cap_n, 384]
        
        # 计算余弦相似度（或点积） -> [query_n, cap_n]
        sim_matrix = torch.matmul(query_feat_norm, global_caption_feat_norm.T)
        labels_tensor = torch.tensor(labels, device=sim_matrix.device)  # [cap_n] list -> tensor
        vid_n = labels_tensor.max().item() + 1  # 视频数量
        query_vid_scores = torch.full((sim_matrix.size(0), vid_n), float('-inf'), device=sim_matrix.device) # [query_n, vid_n]
        query_vid_scores_global = torch.full((sim_matrix.size(0), vid_n), float('-inf'), device=sim_matrix.device) # [query_n, vid_n]
        
        for vid in range(vid_n):
            # 找出属于这个 video 的 caption 的 index
            """
            若: 50个query, 第一个视频有两个caption
            1.提取sim_matrix前两列 [50, 2]
            2.在2这一维度上取最大 [50]
            3.填到目标tensor query_vid_scores 的第一列
            """
            cap_indices = (labels_tensor == vid).nonzero(as_tuple=True)[0]  # 找出第 i 个 video 的 caption 的位置
            if cap_indices.numel() == 0:
                continue  # 某些 video 可能没有 caption
            
            sim_sub = sim_matrix[:, cap_indices] # [query_n, num_caps_for_vid]
            max_sim = sim_sub[:,:-1].max(dim=1).values # 沿着 cap 维度最大 → [query_n]
            query_vid_scores[:, vid] = local_weight * max_sim + (1-local_weight) * sim_sub[:,-1] # 填入第 vid 列

            query_vid_scores_global[:, vid] = sim_sub[:,-1]
            
        return query_feat_norm,query_vid_scores, query_vid_scores_global # shape: [query_n, vid_n]

def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)
