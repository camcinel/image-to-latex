import torch
import torchvision
import math
import torch.nn as nn
from position_encoding import WordPositionalEncoding, ImagePositionalEncoding

# Some code adapted from https://github.com/kingyiusuen/image-to-latex.
# We changed 2D position encoding and some layers of the network.

# define the encoder using a ResNet-18 backbone
class Encoder(nn.Module):
    def __init__(self, d_model=256):
        super(Encoder, self).__init__()
        conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet = torchvision.models.resnet18(pretrained=False)
        self.resnet = nn.Sequential(
            conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )
        self.bottleneck = nn.Conv2d(256, d_model//2, 1)
        self.image_pos_encoder = ImagePositionalEncoding(d_model)
        #self.image_pos_encoder = ImagePositionalEncoding(64)

    def forward(self, x):
        #x = self.conv1(x)
        #x = self.image_pos_encoder(x)
        x = self.resnet(x)
        x = self.bottleneck(x)
        x = self.image_pos_encoder( torch.cat( (x, x), dim=1 ) )
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)
        return x

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



class MathEquationConverter_1(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dim_forward, dropout, num_classes, max_len):
        super(MathEquationConverter_1, self).__init__()
        self.encoder = Encoder(d_model)
        self.decoder = Decoder(d_model, num_heads, num_layers, dim_forward, dropout, num_classes, max_len)
        self.max_len = max_len

    def forward(self, x, y):
        # pass the input through the encoder
        x = self.encoder(x)

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

        # early stopping code is adapted from https://github.com/kingyiusuen/image-to-latex.
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
