import torch
import math
from torch import nn, Tensor
from einops.layers.torch import Rearrange
from ..utils.position_encoding import ImagePositionalEncoding, WordPositionalEncoding

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
        super().__init__()
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
        super().__init__()
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