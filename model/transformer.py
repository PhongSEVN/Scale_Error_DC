import torch
import torch.nn as nn


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model: int = 256, nhead: int = 8,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, src: torch.Tensor, pos: torch.Tensor,
                src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        q = k = src + pos
        attn_output, _ = self.self_attn(q, k, src, key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout1(attn_output))
        src = self.norm2(src + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(src))))))
        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model: int = 256, nhead: int = 8,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                pos: torch.Tensor, query_pos: torch.Tensor,
                memory_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        q = k = tgt + query_pos
        attn_output, _ = self.self_attn(q, k, tgt)
        tgt = self.norm1(tgt + self.dropout1(attn_output))

        attn_output, _ = self.multihead_attn(
            tgt + query_pos, memory + pos, memory,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = self.norm2(tgt + self.dropout2(attn_output))
        tgt = self.norm3(tgt + self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(tgt))))))
        return tgt


class DETRTransformer(nn.Module):

    def __init__(self, d_model: int = 256, nhead: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, pos: torch.Tensor,
                query_embed: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B = src.shape[1]

        memory = src
        for layer in self.encoder_layers:
            memory = layer(memory, pos, src_key_padding_mask=mask)

        N = query_embed.shape[0]
        tgt = torch.zeros(N, B, self.d_model, device=src.device)
        query_pos = query_embed.unsqueeze(1).repeat(1, B, 1)

        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, pos, query_pos, memory_key_padding_mask=mask)

        return tgt
