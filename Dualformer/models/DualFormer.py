

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_patch
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class Model(nn.Module):


    def __init__(self, config):
        super().__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_ff = config.d_ff
        self.factor = config.factor
        self.dropout = config.dropout
        self.activation = config.activation
        self.short_layers = config.short_layers
        self.long_layers = config.long_layers

        # ==================== Dual Patch + Embedding ====================

        self.patch_embedding = DataEmbedding_patch(
            patch_len_list=[4, 12],
            d_model=self.d_model,
            dropout=self.dropout
        )

        # ==================== Dual Expert Layers ====================
        def make_expert_layer():
            attn = AttentionLayer(
                FullAttention(mask_flag=False,
                                factor=self.factor,
                                attention_dropout=self.dropout,
                                output_attention=False),
                self.d_model,
                self.n_heads
            )
            act = nn.GELU() if self.activation == 'gelu' else nn.ReLU()
            ffn = nn.Sequential(
                nn.Linear(self.d_model, self.d_ff),
                act,
                nn.Dropout(self.dropout),
                nn.Linear(self.d_ff, self.d_model)
            )
            return nn.ModuleDict({
                'attn': attn,
                'ffn': ffn,
                'norm1': nn.LayerNorm(self.d_model),
                'norm2': nn.LayerNorm(self.d_model)
            })

        self.short_expert = nn.ModuleList([make_expert_layer() for _ in range(self.short_layers)])
        self.long_expert  = nn.ModuleList([make_expert_layer() for _ in range(self.long_layers)])


        max_layers = max(self.short_layers, self.long_layers)
        self.gating = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model * 2, 2),
                nn.Softmax(dim=-1)
            ) for _ in range(max_layers)
        ])


        self.n_patch_short = self._get_patch_num(config.seq_len, p_len=4, stride=2)
        self.head = nn.Linear(self.n_patch_short * self.d_model, self.pred_len)
        self.norm = nn.LayerNorm(self.d_model)


    def _get_patch_num(self, seq_len, p_len, stride):

        return (seq_len - p_len) // stride + 1

    def _patch_value(self, x, p_len, stride):

        x = x.squeeze(-1)  # [B, L]
        L = x.shape[1]
        pad_len = (stride - (L - p_len) % stride) % stride
        if pad_len:
            x = F.pad(x, (0, pad_len))
        return x.unfold(1, p_len, stride)  # [B, N, p_len]


    def _patch_time(self, mark, p_len, stride):
        if mark is None:
            return None
        # mark: [B, L, 3]
        L = mark.shape[1]
        pad_len = (stride - (L - p_len) % stride) % stride
        if pad_len:
            mark = F.pad(mark, (0, 0, 0, pad_len))  # pad seq dim
        patched = mark.unfold(1, p_len, stride)  # [B, N, 3, p_len]
        return patched.permute(0, 1, 3, 2).contiguous()  # [B, N, p_len, 3]

    # ====================== Forward ======================
    def forward(self, x_enc, x_mark_enc=None, type_id=None):

        if x_enc.dim() == 2:
            x_enc = x_enc.unsqueeze(-1)      # [B, L, 1]

        B, L, _ = x_enc.shape

        # ==================== Dual Branch Patching ====================
        # Short branch (local)
        x_short_val = self._patch_value(x_enc, p_len=4,  stride=2)          # [B, N_short, 4]
        x_short_mark = self._patch_time(x_mark_enc, p_len=4, stride=2)     # [B, N_short, 4, 3]

        # Long branch (global)
        x_long_val = self._patch_value(x_enc, p_len=12, stride=6)          # [B, N_long, 12]
        x_long_mark = self._patch_time(x_mark_enc, p_len=12, stride=6)     # [B, N_long, 12, 3]

        # ==================== Dual Embedding ====================
        x_short = self.patch_embedding(x_short_val, x_short_mark, type_id, branch_idx=0)  # [B, N_short, d_model]
        x_long  = self.patch_embedding(x_long_val,  x_long_mark,  type_id, branch_idx=1)   # [B, N_long,  d_model]

        if x_long.size(1) != x_short.size(1):
            x_long = F.interpolate(
                x_long.transpose(1, 2),                 # [B, d_model, N_long]
                size=x_short.size(1),                   # target N_short
                mode='linear',
                align_corners=False
            ).transpose(1, 2)                           # [B, N_short, d_model]

        for i in range(max(self.short_layers, self.long_layers)):
            # ---- Short Expert ----
            if i < self.short_layers:
                layer = self.short_expert[i]
                x_short = layer['norm1'](x_short)
                attn_out, _ = layer['attn'](x_short, x_short, x_short, attn_mask=None)
                x_short = x_short + attn_out
                x_short = x_short + layer['ffn'](layer['norm2'](x_short))


            if i < self.long_layers:
                layer = self.long_expert[i]
                x_long = layer['norm1'](x_long)
                attn_out, _ = layer['attn'](x_long, x_long, x_long, attn_mask=None)
                x_long = x_long + attn_out
                x_long = x_long + layer['ffn'](layer['norm2'](x_long))


            if i < len(self.gating):
                combined = torch.cat([x_short, x_long], dim=-1)      # [B, N, 2*d_model]
                gate = self.gating[i](combined)                     # [B, N, 2]
                x_short = gate[..., 0:1] * x_short + gate[..., 1:2] * x_long

        # ==================== output ====================
        x = self.norm(x_short)                     # [B, N_short, d_model]
        out = self.head(x.flatten(1))              # [B, pred_len]
        return out