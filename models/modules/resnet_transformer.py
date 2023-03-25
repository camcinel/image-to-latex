import torch
import torchvision
import math
import torch.nn as nn
from ..utils.position_encoding import WordPositionalEncoding, ImagePositionalEncoding

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



class MathEquationConverter(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dim_forward, dropout, num_classes, max_len):
        super().__init__()
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
    
    def predict_beamsearch(self, a, B):  #, B ): # USING BEAM SEARCH
        x = self.encoder(a)
        l = self.max_len - 1
        A = x.size(0)
        
        prev_probs_id = [torch.ones(B)] * x.size(0) # A x B
        prev_probs = torch.stack(prev_probs_id).cuda()

        output_indices = torch.full((B, A, l+1), 0).type_as(x).long()
        output_indices[:, :, 0] = 1
        has_ended = torch.full((A, B,), False)
        
        for pos in range(l): # for each position in a given caption
            N=pos+1

            all_paths_probs = torch.ones(x.size(0),B,B) #init probability tensor
            all_paths_words = torch.zeros(x.size(0),B,B) #init word-idx tensor

            for b in range(B):
                
                y = output_indices[b][:, :N]  # (A, pos)
                logitsb = self.decoder(x, y)  # (A, pos, num_classes) # log of probs (workable numbers)
                preds_sort, ind_sort = torch.sort(logitsb,dim=2,descending=True)

                top_pred_words = ind_sort[:,:,:B] # get the top B, where B is the beam width
                top_pred_probs = preds_sort[:,:,:B]
                top_pred_words = torch.permute(top_pred_words, (0, 2, 1))
                top_pred_probs = torch.permute(top_pred_probs, (0, 2, 1))
                
                all_paths_probs[:,b] = top_pred_probs[:,:,-1] #copy to the BxB matrix with all path probabilities
                all_paths_words[:,b] = top_pred_words[:,:,-1]
            
            flat_add_array = [[[has_ended[batch_idx][i] for i in range(B)] for j in range(B)] for batch_idx in range(A)]
            
            all_paths_add_flat = torch.tensor(flat_add_array).flatten(start_dim=1)
            all_paths_words_flat = all_paths_words.flatten(start_dim=1,end_dim=2)
            all_paths_probs_flat = all_paths_probs.flatten(start_dim=1,end_dim=2)
           
            flatprob_sort, flatidx_sort = torch.sort(all_paths_probs_flat, dim=1, descending=True) #sort all possibilities
            # A x B  
            top_probs = flatprob_sort[:,:B].cuda()
            top_words = torch.gather(all_paths_words_flat, 1, flatidx_sort[:,:B]).int().cuda()
            has_ended = torch.gather(all_paths_add_flat, 1 ,flatidx_sort[:,:B]).cuda()
            
            # UPDATE OUTPUT_INDICES and ADDMORE
            for a_idx in range(A):
                for b_idx in range(B):
                    word = top_words[a_idx][b_idx]
                    isended = has_ended[a_idx][b_idx]
                    if isended:
                        continue
                    else:
                        output_indices[b_idx][a_idx][N] = word
                        
                    if word.item() == 0:
                        has_ended[a_idx][b_idx] = True
            
            ###################################
            

            min_probs = torch.min(prev_probs,dim=1)[0] #get the min value, not the index
            min_probs = torch.unsqueeze(min_probs,1)
            prev_probs = prev_probs - min_probs
            prev_probs = top_probs + prev_probs #update the probabilities in log space
            
            if all(torch.ravel(has_ended)):
                break
        
        maxidx = torch.argmax(prev_probs, dim=1) # pick the best beam
        output_indices = torch.permute(output_indices, (1,0,2))
        output_indices = [output_indices[i][maxidx[i].item()] for i in range(A)]
        return output_indices # [list of A captions, where A is the batch number]