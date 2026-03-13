import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict
from method.model_components import BertAttention, LinearLayer, BertSelfAttention, TrainablePositionalEncoding
from method.model_components import clip_nce, frame_nce
from method.model_loss import LocalHingeLoss



class MS_SL_Net(nn.Module):
    def __init__(self, config):
        super(MS_SL_Net, self).__init__()
        self.config = config
        self.hca_loss_type = self.config.hca_loss_type
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

        self.local_hinge_loss = LocalHingeLoss(margin=self.config.local_margin, reduction='mean', hca_loss_type=self.config.hca_loss_type)
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
        
        clip_scale_scores, frame_scale_scores, clip_scale_scores_, frame_scale_scores_,_,_,query_labels_,_,raw_clip_scale_scores,encoded_query\
            = self.get_pred_from_raw_query(
            query_feat, query_mask, query_labels, vid_proposal_feat, encoded_frame_feat, frame_video_mask, return_query_feats=True, global_caption_mask=global_caption_mask)
        # import pdb;
        # pdb.set_trace()
        
        # clip_scale_scores, frame_scale_scores, key_clip_indices  \
        #     = self.get_pred_from_raw_query(
        #     query_feat, query_mask, query_labels, vid_proposal_feat, encoded_frame_feat, frame_video_mask, return_query_feats=False, global_caption_mask=global_caption_mask)

        raw_label_dict = {}
        for index, label in enumerate(query_labels):
            if label in raw_label_dict:
                raw_label_dict[label].append(index)
            else:
                raw_label_dict[label] = []
                raw_label_dict[label].append(index)
        
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
        
        if self.config.local_hinge_weight > 0.0:
            local_hinge_loss = self.config.local_hinge_weight * self.local_hinge_loss(global_caption_mask,raw_clip_scale_scores,local_start_end_tensor,raw_label_dict)
        else:
            local_hinge_loss = 0

        if self.config.global_soft_pos_weight > 0.0:
            global_soft_pos_loss = self.config.global_soft_pos_weight * self.global_soft_loss(raw_clip_scale_scores, query_labels, global_caption_mask)
        else:
            global_soft_pos_loss = 0
            
        # import ipdb;
        # ipdb.set_trace()

        loss = clip_nce_loss + clip_trip_loss + frame_nce_loss + frame_trip_loss + \
               local_hinge_loss + global_soft_pos_loss

        return loss, {"loss_overall": float(loss), 'clip_nce_loss': clip_nce_loss,
                      'clip_trip_loss': clip_trip_loss,
                      'frame_nce_loss': frame_nce_loss, 'frame_trip_loss': frame_trip_loss,
                      'local_hinge_loss': local_hinge_loss,
                      'global_soft_pos_loss': global_soft_pos_loss}



    def encode_query(self, query_feat, query_mask):
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
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query).squeeze()  # (N, 2 or 1, D)
        if encoded_query.shape[0] == 1:
            modular_queries = modular_queries.unsqueeze(0)
        return modular_queries


    @staticmethod
    def get_clip_scale_scores(modularied_query, context_feat):
        modularied_query = F.normalize(modularied_query, dim=-1)
        context_feat = F.normalize(context_feat, dim=-1)

        clip_level_query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)

        max_clip_scale_context_scores = clip_level_query_context_scores[:, -1, :]  # shape: [Lq ,128]
        
        query_context_scores, indices = torch.max(clip_level_query_context_scores, dim=1)  # (N, N) diagonal positions are positive pairs
        return query_context_scores, indices, max_clip_scale_context_scores, clip_level_query_context_scores



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

        # 新增：如果attention_feat为None或需要替换为ones
        # 假设你需要如下操作：
        # frame_scale_feat = torch.ones(1334,1,384).to(frame_feat.device)
        # 你可以根据实际条件插入
        # 例如：
        # attention_feat = torch.ones(1334,1,384, device=frame_feat.device)
        # 或者在你需要的地方：
        # frame_scale_feat = torch.ones(1334,1,384, device=frame_feat.device)
        # 这里只做演示，具体请根据你的逻辑插入

        return attention_feat



    def get_pred_from_raw_query(self, query_feat, query_mask, query_labels=None,
                                video_proposal_feat=None,
                                video_feat=None,
                                video_feat_mask=None,
                                return_query_feats=False,
                                global_caption_mask=None):


        video_query,encoded_query = self.encode_query(query_feat, query_mask)


        # get clip-level retrieval scores
        clip_scale_scores, key_clip_indices, _, clip_level_query_context_scores = \
            self.get_clip_scale_scores( video_query, video_proposal_feat)
            

        if return_query_feats:
            clip_scale_scores_, _ = self.get_unnormalized_clip_scale_scores(video_query, video_proposal_feat)
            
            raw_clip_scale_scores = clip_level_query_context_scores
            
            if global_caption_mask is not None:
                global_caption_mask_tensor = torch.tensor(global_caption_mask, device=video_query.device)


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
            # import pdb;
            # pdb.set_trace()
            frame_scale_scores_ = torch.matmul(video_query, frame_scale_feat.t())

            return clip_scale_scores, frame_scale_scores, clip_scale_scores_,frame_scale_scores_,_,_,query_labels_,_,raw_clip_scale_scores,encoded_query
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





    def global_soft_loss(self, raw_clip_scale_scores, query_labels, global_caption_mask):
        device = raw_clip_scale_scores.device
        global_caption_mask_tensor = torch.tensor(global_caption_mask, device=device)
        global_indices = (global_caption_mask_tensor == 1).nonzero(as_tuple=True)[0]
        if len(global_indices) == 0:
            return torch.tensor(0.0, device=device)
        # 提取 global_clip_scores
        global_clip_scores = raw_clip_scale_scores[global_indices]  # [num_global, 528, V]
        global_labels = torch.tensor(query_labels, device=device)[global_indices] # [num_global]
        # 正样本分数
        pos_scores = global_clip_scores[:, 527, :]  # [num_global, V]
        pos_scores = pos_scores[torch.arange(len(global_labels)), global_labels]  # [num_global]
        # 150个弱负样本分数
        
        if self.config.window_size == 2:
            weak_scores = global_clip_scores[:, :63, :]  #32+31
            weak_scores = weak_scores[torch.arange(len(global_labels)).unsqueeze(1), torch.arange(63), global_labels.unsqueeze(1)]  # [num_global, 150]
        elif self.config.window_size == 5:
            weak_scores = global_clip_scores[:, :150, :] #32+31+30+29+28
            weak_scores = weak_scores[torch.arange(len(global_labels)).unsqueeze(1), torch.arange(150), global_labels.unsqueeze(1)]  # [num_global, 150]
        elif self.config.window_size == 8:
            weak_scores = global_clip_scores[:, :228, :] #32+31+30+29+28+27+26+25
            weak_scores = weak_scores[torch.arange(len(global_labels)).unsqueeze(1), torch.arange(228), global_labels.unsqueeze(1)]  # [num_global, 150]
        else:
            print("window_size 参数无效")
            exit()
        

        # 下面是t2v的
        if self.hca_loss_type == 'margin':
            loss_batch = torch.clamp(self.config.soft_pos_margin + weak_scores - pos_scores.unsqueeze(1), min=0)
            pos_loss = loss_batch.mean() if loss_batch.numel() > 0 else torch.tensor(0.0, device=device)
        elif self.hca_loss_type == 'infonce':
            # 数值稳定：-(pos - logsumexp(pos ∪ 弱负))
            # 分母仅包含原先选择的弱负样本，加上正样本本身
            # 先构造 [pos, weak_negatives] 的拼接在 logsumexp 空间计算
            # 为了避免构造大张量，这里用等价形式：log(exp(pos) + sum exp(weak))
            denom_log = torch.logsumexp(torch.stack([
                pos_scores,  # shape: [num_global]
                torch.logsumexp(weak_scores, dim=1)  # shape: [num_global]
            ], dim=0), dim=0)
            pos_loss = (denom_log - pos_scores).mean() if pos_scores.numel() > 0 else torch.tensor(0.0, device=device)

        # 新增：local caption与global embedding的hinge loss v2t
        local_indices_all = (global_caption_mask_tensor == 2).nonzero(as_tuple=True)[0]
        total_local_neg_loss = torch.tensor(0.0, device=device)
        num_local_neg = 0
        for i, global_idx in enumerate(global_indices):
            target_video_idx = global_labels[i]
            # global caption的global embedding分数
            pos_score = raw_clip_scale_scores[global_idx, 527, target_video_idx]  # scalar
            # 找到同video的local caption索引
            local_indices = local_indices_all[(torch.tensor(query_labels, device=device)[local_indices_all] == target_video_idx)]
            if len(local_indices) > 0:
                # local caption的global embedding分数
                neg_scores = raw_clip_scale_scores[local_indices, 527, target_video_idx]  # [num_local]
                if self.hca_loss_type == 'margin':
                    loss_local = torch.clamp(self.config.soft_pos_margin + neg_scores - pos_score, min=0)
                elif self.hca_loss_type == 'infonce':
                    # 数值稳定：-(pos - logsumexp(pos ∪ negs))
                    denom_log = torch.logsumexp(torch.cat([neg_scores, pos_score.unsqueeze(0)], dim=0), dim=0)
                    loss_local = denom_log - pos_score
                total_local_neg_loss += torch.sum(loss_local)
                num_local_neg += len(local_indices)
        if num_local_neg > 0:
            local_neg_loss = total_local_neg_loss / num_local_neg
        else:
            local_neg_loss = torch.tensor(0.0, device=device)
        # 返回两个loss之和
        return pos_loss + local_neg_loss
    
    def get_query_global_cap_scores(self,query,query_mask,global_caption_feat):
        query_feat, _ = self.encode_query(query,query_mask)
        # 归一化
        query_feat_norm = F.normalize(query_feat, dim=-1)               # [bsz1, 384]
        global_caption_feat_norm = F.normalize(global_caption_feat, dim=-1)  # [bsz2, 384]

        # 计算余弦相似度（或点积） -> [bsz1, bsz2]
        query_global_cap_scores = torch.matmul(query_feat_norm, global_caption_feat_norm.T)

        return query_global_cap_scores  # 每个 query 与每条 global caption 的相似度分数
    
    
    def key_clip_guided_attention_in_inference_cal_flops(self, frame_feat, proposal_feat, feat_mask, max_index,key,value):
        # key = self.mapping_linear[0](frame_feat)
        # value = self.mapping_linear[1](frame_feat)

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

        # 新增：如果attention_feat为None或需要替换为ones
        # 假设你需要如下操作：
        # frame_scale_feat = torch.ones(1334,1,384).to(frame_feat.device)
        # 你可以根据实际条件插入
        # 例如：
        # attention_feat = torch.ones(1334,1,384, device=frame_feat.device)
        # 或者在你需要的地方：
        # frame_scale_feat = torch.ones(1334,1,384, device=frame_feat.device)
        # 这里只做演示，具体请根据你的逻辑插入

        return attention_feat

    
    def get_pred_from_raw_query_cal_flops(self, query_feat, query_mask, query_labels=None,
                                video_proposal_feat=None,
                                video_feat=None,
                                video_feat_mask=None,
                                return_query_feats=False,
                                global_caption_mask=None,
                                frame_key=None,frame_value=None,top_k=None):

        # 21.9329 MFLOPS
        video_query,encoded_query = self.encode_query(query_feat, query_mask) 
        # video_query = torch.ones(1,384,device=query_feat.device)
        
        
        # #1334 4 384  模拟真实的tt score的计算过程
        a, v, x, b = self.get_clip_scale_scores( video_query, video_proposal_feat[:,:4]) 
        topk = int(video_proposal_feat.shape[0] * top_k)
        video_proposal_feat = video_proposal_feat[:topk]
        video_feat = video_feat[:topk]
        video_feat_mask = video_feat_mask[:topk]
        frame_key = frame_key[:topk]
        frame_value = frame_value[:topk]

        # 540.942 MFLOPS
        # # get clip-level retrieval scores
        # clip_scale_scores, key_clip_indices, _, clip_level_query_context_scores = \
        #     self.get_clip_scale_scores( video_query, video_proposal_feat) 
            

        # clip_scale_scores = torch.ones(1,1334,device=video_proposal_feat.device)
        # key_clip_indices = torch.ones(1,1334,device=video_proposal_feat.device,dtype=torch.long)
        
        
        # 170.752 KFLOPS
        # frame_scale_feat = self.key_clip_guided_attention_in_inference_cal_flops(video_feat, video_proposal_feat, video_feat_mask,key_clip_indices,frame_key,frame_value).to(video_query.device)  
        
        # frame_scale_feat = torch.ones(1334,1,384,device=video_feat.device)


        # # 512.256 KFLOPS
        # frame_scales_cores_ = torch.mul(F.normalize(frame_scale_feat, dim=-1),
        #                                 F.normalize(video_query, dim=-1).unsqueeze(0))
        # frame_scale_scores = torch.sum(frame_scales_cores_, dim=-1).transpose(1, 0)
        
        # frame_scale_scores = torch.ones(1,1334,device=video_proposal_feat.device)

        # return clip_scale_scores, frame_scale_scores, key_clip_indices
        return None, None, None

def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)
