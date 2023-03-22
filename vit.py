import torch
import torchvision
import math
from torch import nn, Tensor
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
import torch.nn.functional as F
import torchvision.models as models


# define the word position encoding layer
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
    """2-D positional encodings for the feature maps produced by the encoder.
    Following https://arxiv.org/abs/2103.06450 by Sumeet Singh.
    Reference:
    https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs/blob/main/lab9/text_recognizer/models/transformer_util.py
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
        #print(f'imageencode: x:{x.shape}, self.pe[:, : x.size(2), : x.size(3)] {self.pe[:, : x.size(2), : x.size(3)].shape}')
        #print(self.pe.shape)
        x[:, :self.d_model//2, :, :] = x[:, :self.d_model//2, :, :] + self.peh[:, : x.size(2), : x.size(3)]
        x[:, self.d_model//2:, :, :] = x[:, self.d_model//2:, :, :] + self.pew[:, : x.size(2), : x.size(3)]
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, 
                 d_model: int = 256,
                 patch_size: int = 16
                ):
        super().__init__()
        
        # For an input image of (1 64 544) 
        # Rearrange (B 1 (4x16) (34*16)) -> (B 1 4 34 (16x16x1))
        # Linear layer (B 1 4 34 (16x16x1)) -> (B 1 4 34 128)
        # Rearrange (B 1 4 34 128) -> (B 128 4 34)
        self.projection = nn.Sequential(Rearrange('b c (h s1) (w s2) -> b h w (s1 s2 c)', s1=patch_size, s2=patch_size),
                                        nn.Linear(patch_size * patch_size, d_model//2),
                                        Rearrange('b h w e -> b e h w')
                                       )
        # Encoding (B 128 4 34) -> (B 256 4 34)
        self.image_pos_enc = ImagePositionalEncoding(d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        x = self.image_pos_enc( torch.cat( (x, x), dim=1 ) )
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)
        return x


class Encoder(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 nhead: int, 
                 num_layers: int, 
                 dim_feedforward: int, 
                 dropout: float, 
                 patch_size: int, 
                 activation: str,
                 batch_first: bool
                ):
        super().__init__()
        # Create single encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, batch_first=True)

        # Create encoder architecture with patch embedding + num_layers*encoder_layer
        self.enc = nn.Sequential(PatchEmbedding(d_model, patch_size),
                                 nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                                )
    
    def forward(self, x):
        return self.enc(x)



# define the transformer decoder
class Decoder(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 nhead: int, 
                 num_layers: int,  
                 dim_feedforward: int,  
                 dropout: float,  
                 num_classes: int, 
                 max_len: int
                ):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.pos_encoder = WordPositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

        # generate the target mask
        mask = torch.tril(torch.ones(max_len, max_len)) == 1
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        self.register_buffer("tgt_mask", mask)

        # create the embedding layer for the target tokens
        self.embedding = nn.Embedding(num_classes, d_model)

    def forward(self, src, tgt):
        # add the positional encoding to the input
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)

        Sy = tgt.shape[1]
        tgt_mask = self.tgt_mask[:Sy, :Sy].type_as(src)

        # use the transformer decoder to decode the input
        output = self.transformer_decoder(tgt, src, tgt_mask)
        output = self.fc(output)

        return output



class MathEquationConverter(nn.Module):
    def __init__(self, 
                 config_encoder: dict, 
                 config_decoder: dict 
                ):
        super(MathEquationConverter, self).__init__()
        self.encoder = Encoder(**config_encoder)
        self.decoder = Decoder(**config_decoder)
        self.max_len = config_decoder['max_len']

    def forward(self, x, y):
        x = self.encoder(x)
        x = self.decoder(x, y)
        return x
    
    def predict(self, x):
        # pass the input through the encoder
        x = self.encoder(x)

        B = x.shape[0]
        S = self.max_len

        output_indices = torch.full((B, S), 0).type_as(x).long()
        output_indices[:, 0] = 1
        has_ended = torch.full((B,), False)

        for Sy in range(1, S):
            y = output_indices[:, :Sy]  # (B, Sy)
            logits = self.decoder(x, y)  # (B, Sy, num_classes)
            # Select the token with the highest conditional probability
            output = torch.argmax(logits, dim=-1)  # (B, Sy)
            output_indices[:, Sy] = output[:, -1]  # Set the last output token
            # Early stopping of prediction loop to speed up prediction
            has_ended |= (output_indices[:, Sy] == 0).type_as(has_ended)
            if torch.all(has_ended):
                break

        return output_indices