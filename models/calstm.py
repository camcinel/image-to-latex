import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.maxpool(self.tanh(self.conv1(x)))
        x = self.maxpool(self.tanh(self.conv2(x)))
        x = self.maxpool(self.tanh(self.conv3(x)))
        x = self.maxpool(self.tanh(self.conv4(x)))
        x = self.maxpool(self.tanh(self.conv5(x)))

        x = x.permute(0, 2, 3, 1)  # (N, D, H, W) -> (N, H, W, D)
        x = x.view(x.size(0), -1, x.size(-1))  # (N, H, W, D) -> (N, H*W = L, D)

        return x


class InitStateModel(nn.Module):
    def __init__(self, L, D, hidden_size, n_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.mlp = nn.Sequential(
            nn.Linear(L * D, 100),
            nn.Tanh(),
            nn.Linear(100, 2 * n_layers * hidden_size),
            nn.Tanh()
        )

    def forward(self, a):
        a = a.reshape(a.size(0), -1)
        hidden_long = self.mlp(a)
        h, c = torch.split(hidden_long, self.n_layers * self.hidden_size, dim=1)  # (N, n_layers * hidden_size)
        h, c = h.view(h.size(0), self.n_layers, self.hidden_size), c.view(c.size(0), self.n_layers,
                                                                          self.hidden_size)  # (N, n_layers, hidden_size)
        h, c = h.permute(1, 0, 2), c.permute(1, 0, 2)  # (n_layers, N, hidden_size)
        return h.contiguous(), c.contiguous()


class AttentionBlock(nn.Module):
    def __init__(self, D, hidden_size):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(D + hidden_size, 256, bias=True),
            nn.Tanh(),
            nn.Linear(256, 128, bias=True),
            nn.Tanh(),
            nn.Linear(128, 1, bias=True)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, a, h):
        h = h.unsqueeze(1).repeat(1, a.size(1), 1)
        att_input = torch.cat((a, h), dim=2)

        alpha = self.mlp(att_input).squeeze(2)
        alpha = self.softmax(alpha)
        z = (alpha.unsqueeze(2) * a).sum(dim=1)

        return z, alpha


class CALSTM(nn.Module):
    def __init__(self, L, D, hidden_size, embedding_size, n_layers, vocab):
        super().__init__()
        self.vocab = vocab
        self.pad_idx = vocab.word2idx['\pad']

        self.attention = AttentionBlock(D, hidden_size)
        self.lstm = nn.LSTM(input_size=D + embedding_size, hidden_size=hidden_size, num_layers=n_layers,
                            batch_first=True)
        self.embed = nn.Embedding(len(vocab), embedding_size)

    def forward(self, a, hidden, y):
        padding = torch.tensor([self.pad_idx] * y.size(0), dtype=torch.long, device=a.get_device()).view(-1, 1)
        y = torch.cat((padding, y[:, :-1]), dim=1)
        y = self.embed(y)
        hze = None  # concatenation to be fed into final linear layer
        for i in range(y.size(1)):
            h, c = hidden
            z, _ = self.attention(a, h[-1])
            input = torch.cat((z, y[:, i, :]), dim=1)
            output, hidden = self.lstm(input.unsqueeze(1), hidden)
            if hze is None:
                hze = torch.cat((output.squeeze(1), z, y[:, i, :]), dim=1).unsqueeze(1)
            else:
                hze = torch.cat((hze, torch.cat((output.squeeze(1), z, y[:, i, :]), dim=1).unsqueeze(1)), dim=1)
        return hze

    def predict(self, a, hidden, prev_word=None):
        if prev_word is None:
            prev_word = torch.tensor([self.pad_idx] * a.size(0), dtype=torch.long, device=a.get_device()).view(-1, 1)  # initial word is '\\pad'
        prev_word = self.embed(prev_word).squeeze(1)
        h, c = hidden
        z, alpha = self.attention(a, h[-1])
        input = torch.cat((z, prev_word), dim=1)
        output, hidden_t = self.lstm(input.unsqueeze(1), hidden)
        hze = torch.cat((output.squeeze(1), z, prev_word), dim=1).unsqueeze(1)
        return hze, hidden_t, alpha


class Decoder(nn.Module):
    def __init__(self, L, D, hidden_size, embedding_size, n_layers, vocab, temperature, determinism, max_length):
        super().__init__()
        self.vocab = vocab
        self.temperature = temperature
        self.determinism = determinism
        self.max_length = max_length

        self.calstm = CALSTM(L, D, hidden_size, embedding_size, n_layers, vocab)

        self.deep_output_layer = nn.Sequential(
            nn.Linear(D + hidden_size + embedding_size, 358),
            nn.Tanh(),
            nn.Linear(358, 358),
            nn.Tanh(),
            nn.Linear(358, len(vocab))
        )

    def forward(self, a, hidden, y):
        output = self.calstm(a, hidden, y)
        output = self.deep_output_layer(output)

        return output

    def predict(self, a, hidden):
        prev_word = None
        captions = [[] for _ in range(a.size(0))]
        add_more = [True] * a.size(
            0)  # tells model to add another word. Marked to false when the last word added is '\\end'
        alphas = None

        for _ in range(self.max_length):
            hze, hidden_t, alpha = self.calstm.predict(a, hidden, prev_word=prev_word)
            pred = self.deep_output_layer(hze).squeeze(1)
            pred.div_(self.temperature)
            pred = nn.functional.softmax(pred, dim=1)
            if self.determinism:
                word_idx = torch.argmax(pred, dim=1).view(-1, 1)
            else:
                word_idx = torch.multinomial(pred, 1)

            for i, (caption, idx, should_add) in enumerate(zip(captions, word_idx, add_more)):
                if should_add:
                    caption.append(self.vocab.idx2word[idx.item()])
                if idx == self.vocab.word2idx['\eos']:
                    add_more[i] = False

            if alphas is None:
                alphas = alpha.unsqueeze(1)
            else:
                alphas = torch.cat((alphas, alpha.unsqueeze(1)), dim=1)

            if not any(add_more):
                break

            prev_word = word_idx
            hidden = hidden_t
        return captions, alphas
    
    def predict_beamsearch(self, a, hidden, B): # USING BEAM SEARCH
        l = self.max_length #maximum caption output length
        A = a.size(0) # number of batches
        H = hidden[0].size(2) #size of the hidden layers
        
        prev_words = [None] * B  # initialize the previous word list
        alphas = [None]*B  #initialize the self-attention list
        
        prev_probs_id = [torch.zeros(B)] * a.size(0) # log(prob) neutral value is 0.
        prev_probs = torch.stack(prev_probs_id).cuda() # A x B
        
        # **** hidden is a tuple of 2 of tensors with dim [2, A, H]
        beam_deep_hidden = torch.stack( [ torch.stack(hidden) ] * B )
        beam_deep_hidden = torch.permute(beam_deep_hidden, (0, 3 , 1, 2 , 4) )
        # **** beam_deep_hidden is tensor of B x A x 2 x 2 x H # where 2 is number of hidden layers.
        
        old_captions = [] #initialize captions list
        add_more = [] #initialize the list that tracks which beams of which batches should add more.

        for a_idx in range(A): # initializing the captions and add_more word tracker lists
            caption = []
            should_add = []
            for b in range(B):
                caption.append([])
                should_add.append(True)
            old_captions.append(caption)
            add_more.append(should_add)
        # ^ this is important bc useing the []*number syntax
        #  is bad bc then each list will have same memory address and not update properly


        for pos in range(l): # for each position in a given caption
            N = pos+1
            # B x A x n x 136
            all_paths_probs = torch.ones(a.size(0),B,B) #init probability tensor
            all_paths_words = torch.zeros(a.size(0),B,B) #init word-idx tensor
            all_paths_hidden = torch.zeros(B, B, A, 2, 2, H)
            
            all_paths_alphas = torch.zeros(B, B, A, N, a.size(1))
            

            for b in range(B):
                
                hidden_temp = beam_deep_hidden[b] # [A x 2 x 2 x H]
                hidden_temp = torch.permute(hidden_temp, (1 ,2 ,0 , 3)).contiguous() # [2 x 2 x A x H]
                hidden_temp = (hidden_temp[0], hidden_temp[1])
                # get the hidden layer in recognizable format for input into calstm.predict()

                hze, hidden_new, alpha = self.calstm.predict(a, hidden_temp, prev_word=prev_words[b])
                # alpha size: tensor [A, 136] 
                
                # hidden new is 2-tuple of 2 x A x H tensors
                hidden_new_tensor = torch.stack(hidden_new) #stack the tuples into tensor format
                
                hidden_new = torch.permute(hidden_new_tensor, (2, 0 ,1 ,3))
                # hidden new is A x 2 x 2 x H
                
                all_paths_hidden[:,b]= torch.stack([hidden_new]*B) #make B copies so each path from a hidden state has the 
                # same hidden state "memory"
            
            
                pred = self.deep_output_layer(hze).squeeze(1) #use prediction using prev words to make new pred.
                pred.div_(self.temperature) # apply temp if not deterministic
                pred = nn.functional.softmax(pred, dim=1) # make into probabilities (sum to 1)
                pred = torch.log(pred) # get the log probabilities for each word in vocab (to add exps instead of multiply small numbers)
                
                
                preds_sort, ind_sort = torch.sort(pred,dim=1,descending=True) # sort the probabilities

                top_pred_words = ind_sort[:,:B] # get the top B, where B is the beam width
                top_pred_probs = preds_sort[:,:B]
                
                all_paths_probs[:,b] = top_pred_probs #copy to the BxB matrix with all path probabilities
                all_paths_words[:,b] = top_pred_words
                
                
                # work with the self-attention alphas to track the same way as hidden layers etc.
                alphas_beam = alphas[b]
                if alphas_beam is None:
                    alphas_beam = alpha.unsqueeze(1)
                else:
                    #alphas_beam = torch.tensor(alphas_beam)
                    alphas_beam = torch.cat((alphas_beam, alpha.unsqueeze(1)), dim=1)

                all_paths_alphas[:,b] = torch.stack([alphas_beam]*B)
                    
                # ***************  
            all_paths_caps = [[[old_captions[batch_idx][i] for i in range(B)] for j in range(B)] for batch_idx in range(A)]
            all_paths_add = [[[add_more[batch_idx][i] for i in range(B)] for j in range(B)] for batch_idx in range(A)]
            
            cap_array = np.array(all_paths_caps , dtype=list)
            flat_array = cap_array.reshape((cap_array.shape[0], B * B, -1))
            
            add_array = np.array(all_paths_add,dtype=object)
            flat_add_array = add_array.reshape((add_array.shape[0], B * B, -1))

            all_paths_cap_flat = flat_array.tolist()        #flatten all captions, words, and probabilities in each batch
            all_paths_add_flat = flat_add_array.tolist()
            all_paths_words_flat = all_paths_words.flatten(start_dim=1)
            all_paths_probs_flat = all_paths_probs.flatten(start_dim=1)
            all_paths_hidden_flat = all_paths_hidden.flatten(start_dim=0,end_dim=1)
            all_paths_alpha_flat = all_paths_alphas.flatten(start_dim=0,end_dim = 1) # B**2 x A x N x a.size(1)
            
            # B^2 x A x 2 x 2 x H
            
            flatprob_sort, flatidx_sort = torch.sort(all_paths_probs_flat, dim=1, descending=True) #sort all possibilities
            # A x B
              
            top_probs = flatprob_sort[:,:B].cuda()
            top_words = torch.gather(all_paths_words_flat, 1, flatidx_sort[:,:B]).int().cuda()
            
            hidden_sort_idx = torch.stack([ torch.stack([ torch.stack([ flatidx_sort[:,:B] ] * 2) ] * 2) ] * H)
            hidden_sort_idx = torch.permute(hidden_sort_idx, (3, 4, 1, 2, 0) )
            hidden_flat_reshape = torch.permute(all_paths_hidden_flat, (1, 0 , 2 ,3, 4))
            #  A x B^2 x 2 x 2 x H
            
            top_hidden = torch.gather(hidden_flat_reshape, dim=1, index = hidden_sort_idx).cuda() # gather best hidden layers
            beam_deep_hidden = torch.permute(top_hidden, (1,0,2,3,4)).contiguous()
            # beam_deep_hidden is tensor of B x A x 2 x 2 x H

            
            alpha_sort_idx = torch.stack( [torch.stack( [ flatidx_sort[:,:B] ] * a.size(1))] * N ) #  N x a.size(1) x A x B
            alpha_sort_idx = torch.permute(alpha_sort_idx, (2, 3, 0, 1) )
            alpha_flat_reshape = torch.permute(all_paths_alpha_flat, (1, 0, 2, 3))
           
            
            top_alpha = torch.gather(alpha_flat_reshape, dim=1, index = alpha_sort_idx).cuda()
            alphas = torch.permute(top_alpha, (1,0,2,3))
            
            
            top_caps = []
            add_more = []
            
            # UPDATE CAPTIONS and ADDMORE (This is not elegant bc we're mixing lists and tensor formats, sorry.)           
            for batch_idx in range(A):
                # we need to do the looping bc of how cloned lists are not new objects and all refer
                # to the same memory address.
                
                beam_caps = []
                beam_adds = []
                beam_words = top_words[batch_idx]
                wd_count=0
                for srtidx in flatidx_sort[batch_idx,:B]:
                    word = beam_words[wd_count]
                    wdstr=self.vocab.idx2word[word.item()]
                    shd_add = all_paths_add_flat[batch_idx][srtidx][0]
                    
                    if pos>0: # if position is not zero
                        if shd_add: # add if we "should add"
                            if type(all_paths_cap_flat[batch_idx][srtidx][0])==list:
                                beam_caps.append( np.append(np.array(all_paths_cap_flat[batch_idx][srtidx][0]),wdstr).tolist())
                            else:
                                beam_caps.append( np.append(np.array(all_paths_cap_flat[batch_idx][srtidx]),wdstr).tolist())
                        else: # don't add anything else, just add what we previously had to the growing list
                            if type(all_paths_cap_flat[batch_idx][srtidx][0])==list:
                                beam_caps.append(all_paths_cap_flat[batch_idx][srtidx][0])
                            else:
                                beam_caps.append(all_paths_cap_flat[batch_idx][srtidx])
                    else: # if position is 0
                        if shd_add:
                            beam_caps.append( [ self.vocab.idx2word[word.item()] ] )
                        else:
                            beam_caps.append([ ])
                    if wdstr=='\\eos': # if the word is the "end of sentance", stop adding to this path.
                        shd_add=False
                        
                    beam_adds.append(shd_add)
                    wd_count+=1 
                top_caps.append(beam_caps)
                add_more.append(beam_adds)

            min_probs = torch.min(prev_probs,dim=1)[0] #get the min value, not the index
            min_probs = torch.unsqueeze(min_probs,1)
            prev_probs = prev_probs - min_probs #dividing by the min value for each batch will not change relative order.
            prev_probs = top_probs + prev_probs #update the probabilities in log space
            
            prev_words = top_words.transpose(0,1).reshape(B,A,1) #so prev_words can fit into the self.calstm.predict funct.
            
            old_captions = top_caps
            
            if not any(np.array(add_more).ravel()):
                break
            
        maxidx = torch.argmax(prev_probs, dim=1)
        caps = [top_caps[i][maxidx[i]] for i in range(A)] # get the best captions per batch
        alphas = torch.permute(alphas, (1,0,2,3))
        alphas_final = [alphas[i][maxidx[i]] for i in range(A)]

        return caps , alphas_final # [list of A best captions, where A is the batch number]


class ImageToLatex(nn.Module):
    def __init__(self, L, D, hidden_size, embedding_size, n_layers, vocab, temperature, determinism, max_length):
        super().__init__()
        self.vocab = vocab

        self.encoder = Encoder()
        self.decoder = Decoder(L, D, hidden_size, embedding_size, n_layers, vocab, temperature, determinism, max_length)
        self.init_state = InitStateModel(L, D, hidden_size, n_layers)

    def forward(self, image, y):
        a = self.encoder(image)
        hidden = self.init_state(a)
        scores = self.decoder(a, hidden, y)

        return scores

    def predict(self, image, beamWidth):
        a = self.encoder(image)
        hidden = self.init_state(a)
        if beamWidth==1:
            captions, alphas = self.decoder.predict(a, hidden)
        elif beamWidth > 1:
            captions, alphas = self.decoder.predict_beamsearch(a, hidden, beamWidth)

        return captions, alphas

    def set_determinism(self, value):
        self.determinism = value
        self.decoder.determinism = value
