import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(MultiHeadedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    
    def forward(self, query_embeds, kv_embeds, query_mask=None, kv_mask=None):
        """
        Input
            query_embeds: batch_size x query_len x embed_dim
            kv_embeds: batch_size x kv_len x embed_dim
            query_mask: batch_size x query_len (optional)
            kv_mask: batch_size x kv_len (optional)
        Output
            o: batch_size x query_len x embed_dim
        """
        batch_size, query_len, _ = query_embeds.shape
        # batch_size x query_len x embed_dim
        q = self.q_proj(query_embeds)
        q = q.reshape(batch_size, query_len, self.num_heads, self.head_dim)
        # batch_size x num_heads x query_len x head_dim
        q = q.permute(0, 2, 1, 3)

        batch_size_kv, kv_len, _ = kv_embeds.shape
        # batch_size_kv x kv_len x embed_dim
        k = self.k_proj(kv_embeds)
        k = k.reshape(batch_size_kv, kv_len, self.num_heads, self.head_dim)
        # batch_size_kv x num_heads x kv_len x head_dim
        k = k.permute(0, 2, 1, 3)

        # batch_size_kv x kv_len x embed_dim
        v = self.v_proj(kv_embeds)
        v = v.reshape(batch_size_kv, kv_len, self.num_heads, self.head_dim)
        # batch_size_kv x num_heads x kv_len x head_dim
        v = v.permute(0, 2, 1, 3)

        # batch_size x num_heads x query_len x kv_len
        attention_logits = q @ k.permute(0, 1, 3, 2)
        attention_logits = attention_logits / math.sqrt(self.head_dim)

        if query_mask is not None:
            # query_mask: batch_size x query_len -> batch_size x 1 x query_len x 1
            attention_logits = attention_logits.masked_fill(query_mask.unsqueeze(1).unsqueeze(-1) == 0, float('-inf'))
        if kv_mask is not None:
            # kv_mask: batch_size x kv_len -> batch_size x 1 x 1 x kv_len
            attention_logits = attention_logits.masked_fill(kv_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        attention_weights = F.softmax(attention_logits, dim=-1)

        # batch_size x num_heads x query_len x head_dim
        attention = attention_weights @ v
        # batch_size x query_len x num_heads x head_dim
        attention = attention.permute(0, 2, 1, 3)
        attention = attention.reshape(batch_size, query_len, self.embed_dim)

        # batch_size x query_len x embed_dim
        o = self.out_proj(attention)
        return o


class Transformer(nn.Module):
    def __init__(self, embed_dim: int, num_mha_heads: int, transformer_dropout: float):
        super(Transformer, self).__init__()
        self.embed_dim = embed_dim
        dropout = transformer_dropout

        self.cross_attn = MultiHeadedAttention(embed_dim, num_mha_heads)

        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)
            
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    
    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)


    def forward(self, text_embeds, global_embeds, text_mask=None, global_mask=None):
        """
        Input
            text_embeds: batch_size x query_len x embed_dim
            global_embeds: batch_size x kv_len x embed_dim
            text_mask: batch_size x query_len (optional)
            global_mask: batch_size x kv_len (optional)
        Output
            out: batch_size x query_len x embed_dim
        """
        text_embeds = self.layer_norm1(text_embeds)
        global_embeds = self.layer_norm1(global_embeds)

        # batch_size x query_len x embed_dim
        attn_out = self.cross_attn(text_embeds, global_embeds, text_mask, global_mask)
        attn_out = self.layer_norm2(attn_out)

        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)

        return out

if __name__ == "__main__":
    # Removed MockConfig and direct config object creation
    embed_dim = 384
    num_mha_heads = 4
    transformer_dropout = 0.1

    batch_size = 2
    query_len = 10
    kv_len = 20

    text_embeds = torch.randn(batch_size, query_len, embed_dim)
    global_embeds = torch.randn(batch_size, kv_len, embed_dim)

    # Creating masks (optional)
    text_mask = torch.ones(batch_size, query_len).bool()
    global_mask = torch.ones(batch_size, kv_len).bool()

    transformer_model = Transformer(embed_dim, num_mha_heads, transformer_dropout)

    output = transformer_model(text_embeds, global_embeds, text_mask, global_mask)

    print("Transformer output shape:", output.shape)
    assert output.shape == (batch_size, query_len, embed_dim)
    print("Transformer test passed!")
