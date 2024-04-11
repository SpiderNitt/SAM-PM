import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import xformers.ops as xops

class PropagationModule(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.max_num_obj = cfg.dataset.max_num_obj
        self.num_frames = cfg.dataset.num_frames
        self.num_tokens = cfg.num_tokens

        self.pos_embed_wt_tfmm = nn.Parameter(torch.zeros(256))
        self.pos_embed_wt_mpam = nn.Parameter(torch.zeros(256))
        
        self.tfmm = TFMM(input_dim=256, embed_dim=128, num_heads=1, dropout=0.1)
        self.values_mlp = MLP(input_dim=256, hidden_dim=512, dropout=0.25)

        self.mpam = MPAM(input_dim=512, embed_dim=128, dropout=0.1)

        self.dense_embedding_linear = MLP(input_dim=256, hidden_dim=512, dropout=0.25)

        self.layer_norm_input = nn.LayerNorm(256)
        self.layer_norm_values_1 = nn.LayerNorm(256)
        self.layer_norm_mpam = nn.LayerNorm(256)

    def forward(self, embeddings: dict, pos_embed: torch.Tensor) -> torch.Tensor:
        """
        image_embeddings: (B, num_frames=3, 256, 64, 64)
        mask_embeddings: # (B, num_frames=2, P=3, 256, 64, 64)
        gt_mask: (3, H, W)
        prev_masks: (B, num_frames=2, num_obj=3, 256, 256)
        selector: (num_obj=3)
        """
        if "image_embeddings" in embeddings:
            curr_embeddings = embeddings["image_embeddings"][:, -1].permute(0, 2, 3, 1) # (B, 64, 64, 256)
            prev_frames_embeddings = embeddings["image_embeddings"][:, :-1].permute(0, 1, 3, 4, 2) # (B, num_frames=2, 64, 64 256)
        else:
            curr_embeddings = embeddings["current_frame_embeddings"].permute(0, 2, 3, 1)
            prev_frames_embeddings = embeddings["prev_frames_embeddings"].permute(0, 1, 3, 4, 2)
        
        mask_embeddings = embeddings["mask_embeddings"].permute(0, 1, 2, 4, 5, 3) # (B, num_frames=2, num_obj=3, 64, 64, 256)

        pos_embed = pos_embed.permute(0, 2, 3, 1)
        curr_embeddings_0 = curr_embeddings + self.pos_embed_wt_tfmm * pos_embed
        prev_frames_embeddings_0 = prev_frames_embeddings + self.pos_embed_wt_tfmm * pos_embed

        curr_embeddings_0 = self.layer_norm_input(curr_embeddings_0)
        prev_frames_embeddings_0 = self.layer_norm_input(prev_frames_embeddings_0)

        values = self.tfmm(curr_embeddings_0, prev_frames_embeddings_0, mask_embeddings) # (B, num_objects=3, 64, 64, 256)
        values_shortcut = values
        values = self.values_mlp(values)
        values = self.layer_norm_values_1(values + values_shortcut) # (B, num_objects=3, 64, 64, 256)

        curr_embeddings_1 = curr_embeddings + self.pos_embed_wt_mpam * pos_embed
        prev_frames_embeddings_1 = prev_frames_embeddings + self.pos_embed_wt_mpam * pos_embed

        curr_embeddings_1 = self.layer_norm_mpam(curr_embeddings_1)
        prev_frames_embeddings_1 = self.layer_norm_mpam(prev_frames_embeddings_1)

        curr_embeddings_1 = curr_embeddings_1.unsqueeze(1).repeat(1, values.shape[1], 1, 1, 1) # (B, num_objects=3, 64, 64, 256)
        query = torch.cat([curr_embeddings_1, values], dim=-1) # (B, num_objects=3, 64, 64, 512)

        prev_frames_embeddings_1 = prev_frames_embeddings_1.unsqueeze(2).repeat(1, 1, mask_embeddings.shape[2], 1, 1, 1) # (B, num_frames=2, num_objects=3, 64, 64, 256)
        key = torch.cat([prev_frames_embeddings_1, mask_embeddings], dim=-1) # (B, num_frames=2, num_objects=3, 64, 64, 512)

        dense_embeddings = self.mpam(query, key) # (B, num_objects=3, 64, 64, [num_heads * self.head_dim] = 256)
        
        dense_embeddings = self.dense_embedding_linear(dense_embeddings) # (B, num_objects=3, 64, 64, 256)
        sparse_embeddings = torch.empty((*dense_embeddings.shape[:2], 0, 256), device=dense_embeddings.device) # (B, num_objects=3, 1, 256)

        return (
            sparse_embeddings,
            dense_embeddings,
            {
                "pos_embed_wt_tfmm": self.pos_embed_wt_tfmm,
                "pos_embed_wt_mpam": self.pos_embed_wt_mpam,
            },
        )
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TFMM(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.key_head_dim = embed_dim // num_heads
        self.value_head_dim = input_dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(input_dim, embed_dim)
        self.k_proj = nn.Linear(input_dim, embed_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)

        self.o_proj = nn.Linear(input_dim, input_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.q_proj.bias.data.fill_(0)
        self.k_proj.bias.data.fill_(0)
        self.v_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
    
    def forward(self, q, k, v):
        batch_size = q.size(0)
        num_frames = k.size(1)
        num_objects = v.size(2)
        seq_len = 64*64

        q = self.q_proj(q) # (B, 64, 64, embed_dim=256)
        k = self.k_proj(k) # (B, num_frames=2, 64, 64, embed_dim=256)
        v = self.v_proj(v) # (B, num_frames=2, num_objects=3, 64, 64 embed_dim=256)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.key_head_dim)
        k = k.reshape(batch_size, seq_len*num_frames, self.num_heads, self.key_head_dim)
        v = v.permute(0, 1, 3, 4, 5, 2).reshape(batch_size, seq_len*num_frames, self.num_heads, self.value_head_dim*num_objects)

        if self.training:
            values = xops.memory_efficient_attention(q, k, v, p=self.dropout) # (B, 64*64, num_heads, self.head_dim*num_obj=3)
        else:
            values = xops.memory_efficient_attention(q, k, v, p=0) # (B, 64*64, num_heads, self.head_dim*num_obj=3)
        values = values.reshape(batch_size, 64, 64, self.num_heads * self.value_head_dim, num_objects)
        values = values.permute(0, 4, 1, 2, 3) # (B, num_objects=3, 64, 64, [num_heads * self.head_dim] = 256)

        values = self.o_proj(values)
        return values


class MPAM(nn.Module):
    def __init__(self, input_dim, embed_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.qk_proj = nn.Linear(input_dim, embed_dim)
        self.qv_proj = nn.Linear(input_dim, embed_dim)
        self.mk_proj = nn.Linear(input_dim, embed_dim)
        self.mv_proj = nn.Linear(input_dim, embed_dim)
    
    def forward(self, q, m):
        batch_size = q.size(0)
        num_objects = q.size(1)
        num_frames = m.size(1)
        seq_len = 64*64
        
        qk = self.qk_proj(q) # (B, P, 64, 64, embed_dim=128)
        qv = self.qv_proj(q) # (B, P, 64, 64, embed_dim=128)
        mk = self.mk_proj(m) # (B, F, P, 64, 64, embed_dim=128)
        mv = self.mv_proj(m) # (B, F, P, 64, 64 embed_dim=128)

        qk = qk.reshape(batch_size * num_objects, seq_len, self.embed_dim) # (B*P, 64*64, embed_dim=128)

        mk = mk.transpose(1, 2).reshape(batch_size * num_objects, seq_len*num_frames, self.embed_dim) # (B*P, F*64*64, embed_dim=128)
        mv = mv.transpose(1, 2).reshape(batch_size * num_objects, seq_len*num_frames, self.embed_dim) # (B*P, F*64*64, embed_dim=128)

        if self.training:
            values = xops.memory_efficient_attention(qk, mk, mv, p=self.dropout) # (B*P, 64*64, embed_dim=128)
        else:
            values = xops.memory_efficient_attention(qk, mk, mv, p=0) # (B*P, 64*64, embed_dim=128)
        values = values.reshape(batch_size, num_objects, 64, 64, self.embed_dim) # (B, P, 64, 64, embed_dim=128)

        out = torch.cat([qv, values], dim=-1) # (B, P, 64, 64, embed_dim=256)

        return out # (B, P, 64, 64, embed_dim=256)
    
class Memory():
    def __init__ (self, length) -> None:
        self.embed = []
        self.mask = []
        self.score = []
        self.total_size = length
        self.frames_n = []

    def add(self, image_embed, mask, iou):
        if len(self.embed) < self.total_size:
            self.embed.append(image_embed)
            self.mask.append(mask)
            self.score.append(iou)
                
        else:
            idx = 0
            self.score.pop(idx)
            self.embed.pop(idx)
            self.mask.pop(idx)

            self.embed.append(image_embed)
            self.mask.append(mask)
            self.score.append(iou)
        
    def get_embed(self):
        # image_embed: (F, 256, 64, 64)
        return torch.stack(self.embed, dim=0)
    
    def get_prev_mask(self):
        # (F, P, 256, 256)
        return torch.stack(self.mask, dim=0)