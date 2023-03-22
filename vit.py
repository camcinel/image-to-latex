import torch
import torchvision
import math
from torch import nn, Tensor
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
import torch.nn.functional as F
import torchvision.models as models
# torchvision.models.vision_transformer.VisionTransformer


# class ImagePositionalEncoding_vit(nn.Module):
#     """
#     Module for adding position embeddings to a sequence of image patches.
#     """
#     def __init__(self, num_patches, embed_dim, dropout=0.1, max_len=1200):
#         super().__init__()
#         # self.patch_size = patch_size
#         self.embed_dim = embed_dim
#         self.dropout = nn.Dropout(p=dropout)
        
#         # Create learnable parameters for the position embeddings
#         # num_patches = (224 // patch_size) ** 2
#         # self.position_embeddings = nn.Parameter(self.make_pe(embed_dim, num_patches + 1, max_len))
#         self.position_embeddings = self.make_pe(embed_dim, num_patches + 1, max_len)
#         self.device =  torch.device('cuda' if torch.has_cuda else 'cpu')

#     @staticmethod
#     def make_pe(d_model: int, seq_len: int, max_len: int):
#         """Compute positional encoding."""
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term[:d_model])
#         pe[:, 1::2] = torch.cos(position * div_term[:d_model])
#         return pe
    
#     def forward(self, x):
#         """
#         Applies position embeddings to a batch of image patches.

#         Args:
#             x (tensor): Input tensor of shape (batch_size, num_patches, embed_dim).
        
#         Returns:
#             tensor: Output tensor with position embeddings added, of shape (batch_size, num_patches + 1, embed_dim).
#         """
#         batch_size, num_patches, embed_dim = x.shape
        
#         # Add the CLS token at the beginning of the sequence
#         cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
#         cls_token = cls_token.expand(batch_size, -1, -1).to(self.device)
#         x = torch.cat([cls_token, x], dim=1)
        
#         # Add the position embeddings to each patch and the CLS token
#         position_embeddings = self.position_embeddings[:num_patches + 1].unsqueeze(0)
#         position_embeddings = position_embeddings.expand(batch_size, -1, -1).to(self.device)
#         x = x + position_embeddings
        
#         return self.dropout(x)


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
    def __init__(self, patch_size, emb_size, img_size):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(Rearrange('b c (h s1) (w s2) -> b h w (s1 s2 c)', s1=patch_size, s2=patch_size),
                                        nn.Linear(patch_size * patch_size * img_size[0], emb_size//2),
                                        Rearrange('b h w e -> b e h w')
                                       )
        self.image_pos_enc = ImagePositionalEncoding(emb_size)

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape    # (B,E,H,W)
        x = self.projection(x)
        x = self.image_pos_enc( torch.cat( (x, x), dim=1 ) )
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, dim_feedforward: int, dropout: float, patch_size: int, activation_fn: str):
        super(Encoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout, activation_fn, batch_first=True)

        self.enc = nn.Sequential(PatchEmbedding(patch_size, d_model, img_size=(1, 128, 1088)),
                                 nn.TransformerEncoder(encoder_layer, num_layers)
                                )
    
    def forward(self, x):
        return self.enc(x)



# define the transformer decoder
class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dim_forward, dropout, num_classes, max_len):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.pos_encoder = WordPositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, dim_feedforward=dim_forward, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
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
    def __init__(self, config_encoder, config_decoder, num_classes, max_len):
        super(MathEquationConverter, self).__init__()
        self.encoder = Encoder(*(config_encoder.values()))
        self.decoder = Decoder(*(config_decoder.values()), num_classes, max_len)
        self.max_len = max_len

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