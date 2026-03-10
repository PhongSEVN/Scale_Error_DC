"""
DETR Transformer: Standard Encoder-Decoder Architecture
=========================================================
Ref: Section 3.1 of "End-to-End Object Detection with Transformers"

The Transformer follows the standard architecture from "Attention Is All You Need"
(Vaswani et al., 2017). The encoder processes the flattened feature map with
positional encodings, and the decoder attends to the encoder output using N
learnable object queries (default N=100).

Key design choices from the DETR paper:
    - The encoder uses self-attention over the flattened spatial features.
    - The decoder uses self-attention over object queries, then cross-attention
      to the encoder memory.
    - Positional encodings are added at every attention layer (not just the input).
    - Object queries are learnable embeddings, one per detection slot.
"""

import torch
import torch.nn as nn


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder layer with pre-norm or post-norm style.

    Consists of:
        1. Multi-head Self-Attention
        2. Feed-Forward Network (2-layer MLP with ReLU)
        3. LayerNorm + Residual connections

    Ref: DETR uses post-norm (LayerNorm after residual addition).
    """

    def __init__(self, d_model: int = 256, nhead: int = 8,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        """
        Args:
            d_model: Transformer hidden dimension.
            nhead: Number of attention heads.
            dim_feedforward: Hidden dimension of the FFN.
            dropout: Dropout rate.
        """
        super().__init__()

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=False
        )

        # Feed-Forward Network (FFN)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, src: torch.Tensor, pos: torch.Tensor,
                src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for encoder layer.

        Ref: "We add positional encodings to the input of each attention layer."

        Args:
            src: Source sequence, shape (HW, B, d_model)
            pos: Positional encoding, shape (HW, B, d_model)
            src_key_padding_mask: Mask for padded positions, shape (B, HW)

        Returns:
            src: Updated sequence, shape (HW, B, d_model)
        """
        # Self-attention with positional encoding added to Q and K
        # Ref: Positional encodings are added to queries and keys (not values)
        q = k = src + pos
        attn_output, _ = self.self_attn(
            query=q, key=k, value=src,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # Feed-Forward Network
        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ffn_output)
        src = self.norm2(src)

        return src


class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer Decoder layer.

    Consists of:
        1. Multi-head Self-Attention (over object queries)
        2. Multi-head Cross-Attention (queries attend to encoder memory)
        3. Feed-Forward Network
        4. LayerNorm + Residual connections

    Ref: "The decoder follows the standard architecture, transforming N
          embeddings of size d using multi-head self and encoder-decoder
          attention mechanisms." (Section 3.1)
    """

    def __init__(self, d_model: int = 256, nhead: int = 8,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()

        # Self-attention over object queries
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=False
        )

        # Cross-attention: object queries attend to encoder memory
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=False
        )

        # Feed-Forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                pos: torch.Tensor, query_pos: torch.Tensor,
                memory_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for decoder layer.

        Args:
            tgt: Target (object query) sequence, shape (N, B, d_model)
            memory: Encoder output, shape (HW, B, d_model)
            pos: Spatial positional encoding for memory, shape (HW, B, d_model)
            query_pos: Learnable object query positional embeddings, shape (N, B, d_model)
            memory_key_padding_mask: Mask for encoder padding, shape (B, HW)

        Returns:
            tgt: Updated object queries, shape (N, B, d_model)
        """
        # 1. Self-attention among object queries
        # Ref: "Object queries are added to the input of each decoder attention layer"
        q = k = tgt + query_pos
        attn_output, _ = self.self_attn(query=q, key=k, value=tgt)
        tgt = tgt + self.dropout1(attn_output)
        tgt = self.norm1(tgt)

        # 2. Cross-attention: queries attend to spatial encoder memory
        # Query: object query + query_pos, Key: memory + spatial_pos, Value: memory
        attn_output, _ = self.multihead_attn(
            query=tgt + query_pos,
            key=memory + pos,
            value=memory,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(attn_output)
        tgt = self.norm2(tgt)

        # 3. Feed-Forward Network
        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(ffn_output)
        tgt = self.norm3(tgt)

        return tgt


class DETRTransformer(nn.Module):
    """
    Full DETR Transformer: stacked encoder + decoder layers.

    Ref: "Our transformer uses a standard architecture with 6 encoder and
          6 decoder layers with width 256." (Section 4)
    """

    def __init__(self, d_model: int = 256, nhead: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        """
        Args:
            d_model: Hidden dimension of the Transformer.
            nhead: Number of attention heads in each layer.
            num_encoder_layers: Number of encoder layers (default=6).
            num_decoder_layers: Number of decoder layers (default=6).
            dim_feedforward: FFN hidden dimension (default=2048).
            dropout: Dropout rate.
        """
        super().__init__()

        self.d_model = d_model

        # Build encoder stack
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Build decoder stack
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier uniform initialization for all linear and conv parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, pos: torch.Tensor,
                query_embed: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Full forward pass through encoder and decoder.

        Args:
            src: Flattened feature map from backbone, shape (HW, B, d_model)
            pos: Flattened positional encoding, shape (HW, B, d_model)
            query_embed: Learnable object queries, shape (N, d_model)
            mask: Source key padding mask, shape (B, HW). True = padded.

        Returns:
            decoder_output: Shape (N, B, d_model) — one embedding per object query.
        """
        B = src.shape[1]

        # ==================== ENCODER ====================
        memory = src
        for encoder_layer in self.encoder_layers:
            memory = encoder_layer(memory, pos, src_key_padding_mask=mask)

        # ==================== DECODER ====================
        # Initialize target (object queries) as zeros
        # Ref: "The decoder receives as input N learnable positional embeddings
        #        (object queries), which we add to the input of each attention layer."
        N = query_embed.shape[0]
        tgt = torch.zeros(N, B, self.d_model, device=src.device)

        # Expand query_embed for batch dimension: (N, d_model) -> (N, B, d_model)
        query_pos = query_embed.unsqueeze(1).repeat(1, B, 1)

        for decoder_layer in self.decoder_layers:
            tgt = decoder_layer(
                tgt=tgt,
                memory=memory,
                pos=pos,
                query_pos=query_pos,
                memory_key_padding_mask=mask
            )

        return tgt  # (N, B, d_model)
