# layers/DualEmbed.py
import torch
import torch.nn as nn
import math


class TypeEmbedding(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.emb = nn.Embedding(2, d_model)      # 2 types

    def forward(self, x):
        return self.emb(x.long())                # [B, d_model]


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                     # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1), :]


class DataEmbedding_patch(nn.Module):
    def __init__(self, patch_len_list=[4, 12], d_model=256, dropout=0.1):
        super().__init__()

        self.value_projs = nn.ModuleList([
            nn.Linear(p_len, d_model) for p_len in patch_len_list
        ])

        self.time_projs = nn.ModuleList([
            nn.Linear(3, d_model) for _ in patch_len_list
        ])

        self.pos_emb   = PositionalEmbedding(d_model)
        self.type_emb  = TypeEmbedding(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x_patched, x_mark_patched, type_id, branch_idx=0):

        B, N, p_len = x_patched.shape

        # ==================== 1. Value Embedding ====================
        value_flat = x_patched.reshape(B * N, p_len)
        value_token = self.value_projs[branch_idx](value_flat)           # [B*N, d_model]
        value_token = value_token.view(B, N, -1)                            # [B, N, d_model]

        # ==================== 2. Time Feature Embedding ====================
        time_token = torch.zeros_like(value_token)
        if x_mark_patched is not None:
            time_feat = x_mark_patched.mean(dim=2)                         # [B, N, 3]
            time_flat = time_feat.reshape(B * N, 3)
            time_proj = self.time_projs[branch_idx](time_flat)             # [B*N, d_model]
            time_token = time_proj.view(B, N, -1)                          # [B, N, d_model]

        # ==================== 3. Position + Type Embedding ====================
        pos_token  = self.pos_emb(value_token)                             # [B, N, d_model]
        type_token = self.type_emb(type_id).unsqueeze(1).expand(-1, N, -1)  # [B, N, d_model]

        # ==================== 4. mix ====================
        x = value_token + time_token + pos_token + type_token
        return self.dropout(x)