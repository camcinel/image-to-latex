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

    def __init__(self, d_model: int, max_h: int = 2000, max_w: int = 2000) -> None:
        super().__init__()
        self.d_model = d_model
        assert d_model % 2 == 0, f"Embedding depth {d_model} is not even"
        pe = self.make_pe(d_model, max_h, max_w)  # (d_model, max_h, max_w)
        self.register_buffer("pe", pe)

    @staticmethod
    def make_pe(d_model: int, max_h: int, max_w: int):
        """Compute positional encoding."""
        pe_h = WordPositionalEncoding.make_pe(d_model=d_model // 2, max_len=max_h)  # (1, max_h, d_model // 2)
        pe_h = pe_h.permute(2, 1, 0).expand(-1, -1, max_w)  # (d_model // 2, max_h, max_w)

        pe_w = WordPositionalEncoding.make_pe(d_model=d_model // 2, max_len=max_w)  # (1, max_w, d_model // 2)
        pe_w = pe_w.permute(2, 0, 1).expand(-1, max_h, -1)  # (d_model // 2, max_h, max_w)

        pe = torch.cat([pe_h, pe_w], dim=0)  # (d_model, max_h, max_w)
        # print(f'pe:{pe.shape}')
        return pe

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, d_model, H, W)

        Returns:
            (B, d_model, H, W)
        """
        #print(f'imageencode: x:{x.shape}, self.pe[:, : x.size(2), : x.size(3)] {self.pe[:, : x.size(2), : x.size(3)].shape}')
        #print(self.pe.shape)
        x = x + self.pe[:, : x.size(2), : x.size(3)]
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, emb_size, img_size):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(img_size[0], emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e')
        )
                
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size[1] * img_size[2] // (patch_size**2)) + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, att_dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(att_dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out
    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, forward_expansion, forward_drop_p):
        super().__init__(
            nn.Linear(emb_size, forward_expansion * emb_size),
            nn.GELU(),
            nn.Dropout(forward_drop_p),
            nn.Linear(forward_expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 drop_p,
                 forward_expansion,
                 num_heads,
                ):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, forward_expansion, drop_p),
                nn.Dropout(drop_p)
            )
            ))
        

class Transformer_Encoder(nn.Sequential):
    def __init__(self, depth, emb_size, drop_p, forward_expansion, num_heads):
        super().__init__(*[TransformerEncoderBlock(emb_size, drop_p, forward_expansion, num_heads) for _ in range(depth)])

class ViT(nn.Sequential):
    def __init__(self, emb_size, num_heads, num_layers, dim_forward, dropout, patch_size, **kwargs):

        super().__init__(
            PatchEmbedding(patch_size, emb_size, img_size=(1,256,1152)),
            Transformer_Encoder(num_layers, emb_size, dropout, dim_forward, num_heads)
        )

class VisionConvolutions(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = nn.Sequential(
            conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2         # (128, 32, 144)
            # resnet.layer3,      # (256,8,68)
        )
        # self.bottleneck = nn.Conv2d(256, d_model//2, 1)

    def forward(self, x):
        # x = self.resnet(x)
        return self.resnet(x)


class Encoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, dim_feedforward: int, dropout: float, patch_size: int, activation_fn: str):
        super(Encoder, self).__init__()
        # self.d_model = d_model
        # self.vision_convolutions = VisionConvolutions()
        # self.patch_emb = PatchEmbedding(patch_size, emb_size, img_size=(128, 32, 144))
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout, activation_fn, batch_first=True)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.enc = nn.Sequential(VisionConvolutions(),
                                PatchEmbedding(patch_size, d_model, img_size=(128, 32, 144)),
                                nn.TransformerEncoder(encoder_layer, num_layers)
                                )
    
    def forward(self, x):
        # print(x)
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
        # src = self.pos_encoder(src)
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
        # print(config_decoder)
        super(MathEquationConverter, self).__init__()
        self.encoder = ViT(*(config_encoder.values()))
        self.encoder = Encoder(*(config_encoder.values()))
        self.decoder = Decoder(*(config_decoder.values()), num_classes, max_len)
        self.max_len = max_len

    def forward(self, x, y):
        # pass the input through the encoder
        x = self.encoder(x)
        # print("Encoder output: ", x.size())
        # pass the output of the encoder through the decoder
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