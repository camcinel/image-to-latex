import torch
import torchvision
import math
import torch.nn as nn

class WordPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = self.make_pe(d_model, max_len)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    @staticmethod
    def make_pe(d_model: int, max_len: int):
        """Compute positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        return pe
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    

class ImagePositionalEncoding(nn.Module):
    """
    Following https://arxiv.org/abs/2103.06450 by Sumeet Singh.
    Reference:
    https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs/blob/main/lab9/text_recognizer/models/transformer_util.py
    We changed the implementation to reduce the number of parameters.
    """

    def __init__(self, d_model: int, max_h: int = 1200, max_w: int = 1200) -> None:
        super().__init__()
        self.d_model = d_model
        assert d_model % 2 == 0, f"Embedding depth {d_model} is not even"
        peh, pew = self.make_pe(d_model, max_h, max_w)  # (d_model, max_h, max_w)
        self.register_buffer("peh", peh)
        self.register_buffer("pew", pew)

    @staticmethod
    def make_pe(d_model: int, max_h: int, max_w: int):
        """Compute positional encoding."""
        pe_h = WordPositionalEncoding.make_pe(d_model=d_model // 2, max_len=max_h)  # (1, max_h, d_model // 2)
        pe_h = pe_h.permute(2, 1, 0)  # (d_model // 2, max_h, 1)

        pe_w = WordPositionalEncoding.make_pe(d_model=d_model // 2, max_len=max_w)  # (1, max_w, d_model // 2)
        pe_w = pe_w.permute(2, 0, 1)  # (d_model // 2, 1, max_w)

        return pe_h, pe_w

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, d_model, H, W)

        Returns:
            (B, d_model, H, W)
        """
        x[:, :self.d_model//2, :, :] = x[:, :self.d_model//2, :, :] + self.peh[:, : x.size(2), : x.size(3)]
        x[:, self.d_model//2:, :, :] = x[:, self.d_model//2:, :, :] + self.pew[:, : x.size(2), : x.size(3)]
        return x