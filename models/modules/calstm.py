import torch
import torch.nn as nn
import numpy as np


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
            prev_word = torch.tensor([self.pad_idx] * a.size(0), dtype=torch.long, device=a.get_device()).view(-1,
                                                                                                               1)  # initial word is '\\pad'

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

    def predict_beam(self, a, hidden, B):  # USING BEAM SEARCH
        l = self.max_length
        A = a.size(0)
        prev_words = [None] * B
        prev_probs_id = [torch.ones(B)] * a.size(0)
        prev_probs = torch.stack(prev_probs_id).cuda()

        old_captions = []
        add_more = []
        for _ in range(A):  # initializing the captions and add_more word tracker lists
            caption = []
            should_add = []
            for b in range(B):
                caption.append([])
                should_add.append(True)
            old_captions.append(caption)
            add_more.append(should_add)

        for pos in range(l):  # for each position in a given caption

            print('POS:', pos)
            print('old:', old_captions)
            print('more:', add_more)
            print('prev_words:', prev_words)
            if pos > 0:
                print('last word', [[old_captions[i][b][-1] for b in range(B)] for i in range(A)])

            # print('POSITION:', pos)
            all_paths_probs = torch.ones(a.size(0), B, B)  # init probability tensor
            all_paths_words = torch.zeros(a.size(0), B, B)  # init word-idx tensor

            for b in range(B):
                hze, hidden, _ = self.calstm.predict(a, hidden, prev_word=prev_words[b])
                pred = self.deep_output_layer(hze).squeeze(1)  # use pre
                pred.div_(self.temperature)  # we are semi-deterministic, do we need this here?
                pred = nn.functional.softmax(pred, dim=1)
                pred = torch.log(pred)  # get the log probabilities for each word in vocab
                preds_sort, ind_sort = torch.sort(pred, dim=1, descending=True)  # sort the probabilities

                top_pred_words = ind_sort[:, :B]  # get the top B, where B is the beam width
                top_pred_probs = preds_sort[:, :B]

                all_paths_probs[:, b] = top_pred_probs  # copy to the BxB matrix with all path probabilities
                all_paths_words[:, b] = top_pred_words

            all_paths_caps = [[[old_captions[batch_idx][i] for i in range(B)] for j in range(B)] for batch_idx in
                              range(A)]
            all_paths_add = [[[add_more[batch_idx][i] for i in range(B)] for j in range(B)] for batch_idx in range(A)]

            cap_array = np.array(all_paths_caps, dtype=list)
            flat_array = cap_array.reshape((cap_array.shape[0], B * B, -1))

            add_array = np.array(all_paths_add, dtype=object)
            flat_add_array = add_array.reshape((add_array.shape[0], B * B, -1))

            all_paths_cap_flat = flat_array.tolist()  # flatten all captions, words, and probabilities in each batch

            all_paths_add_flat = flat_add_array.tolist()
            all_paths_words_flat = all_paths_words.flatten(start_dim=1)
            all_paths_probs_flat = all_paths_probs.flatten(start_dim=1)

            # print('flat_array.tolist(): \n', all_paths_cap_flat)

            flatprob_sort, flatidx_sort = torch.sort(all_paths_probs_flat, dim=1,
                                                     descending=True)  # sort all possibilities

            top_probs = flatprob_sort[:, :B].cuda()
            top_words = torch.gather(all_paths_words_flat, 1, flatidx_sort[:, :B]).int().cuda()

            top_caps = []
            add_more = []

            for batch_idx in range(A):

                beam_caps = []
                beam_adds = []
                beam_words = top_words[batch_idx]
                wd_count = 0
                for srtidx in flatidx_sort[batch_idx, :B]:
                    word = beam_words[wd_count]
                    wdstr = self.vocab.idx2word[word.item()]
                    shd_add = all_paths_add_flat[batch_idx][srtidx][0]
                    # print(word.item(),shd_add,all_paths_cap_flat[batch_idx][srtidx])

                    if pos > 0:
                        if shd_add:
                            if type(all_paths_cap_flat[batch_idx][srtidx][0]) == list:
                                # print('print beam cap 1', all_paths_cap_flat[batch_idx][srtidx][0])
                                beam_caps.append(
                                    np.append(np.array(all_paths_cap_flat[batch_idx][srtidx][0]), wdstr).tolist())
                            else:
                                # print('print beam cap 2', all_paths_cap_flat[batch_idx][srtidx])
                                # print('to arr to lst', np.append(np.array(all_paths_cap_flat[batch_idx][srtidx]),'3').tolist() )
                                beam_caps.append(
                                    np.append(np.array(all_paths_cap_flat[batch_idx][srtidx]), wdstr).tolist())
                        else:
                            if type(all_paths_cap_flat[batch_idx][srtidx][0]) == list:
                                # print('print beam cap 3', all_paths_cap_flat[batch_idx][srtidx][0])
                                beam_caps.append(all_paths_cap_flat[batch_idx][srtidx][0])
                            else:
                                # print('print beam cap 4', all_paths_cap_flat[batch_idx][srtidx])
                                beam_caps.append(all_paths_cap_flat[batch_idx][srtidx])
                    else:
                        if shd_add:
                            beam_caps.append([self.vocab.idx2word[word.item()]])
                            # beam_caps.append(all_paths_cap_flat[batch_idx][srtidx].append( self.vocab.idx2word[word.item()] ))
                        else:
                            beam_caps.append([])

                        # beam_caps.append(all_paths_cap_flat[batch_idx][srtidx])
                    if wdstr == '\\eos':
                        shd_add = False

                    beam_adds.append(shd_add)
                    wd_count += 1
                    # print('beam_caps:', beam_caps)
                # print('beam_adds:', beam_adds)
                top_caps.append(beam_caps)
                add_more.append(beam_adds)

            # top_caps = [ [all_paths_cap_flat[batch_idx][srtidx] for srtidx in flatidx_sort[batch_idx,:B]] for batch_idx in range(A)]
            # add_more = [ [all_paths_add_flat[batch_idx][srtidx][0] for srtidx in flatidx_sort[batch_idx,:B]] for batch_idx in range(A)]

            min_probs = torch.min(prev_probs, dim=1)[0]  # get the min value, not the index
            min_probs = torch.unsqueeze(min_probs, 1)
            prev_probs = prev_probs - min_probs
            prev_probs = top_probs + prev_probs  # update the probabilities in log space

            prev_words = top_words.transpose(0, 1).reshape(B, A,
                                                           1)  # so prev_words can fit into the self.calstm.predict funct.

            old_captions = top_caps

            if not any(np.array(add_more).ravel()):
                # print('NO MORE POSITIONS')
                break

        maxidx = torch.argmax(prev_probs, dim=1)
        caps = [top_caps[i][maxidx[i]] for i in range(A)]  # get the best captions per batch
        # print(caps)
        return caps  # [list of A captions, where A is the batch number]

    def predict(self, a, hidden):
        prev_word = None
        captions = []
        for _ in range(a.size(0)):
            captions.append([])
        add_more = [True] * a.size(
            0)  # tells model to add another word. Marked to false when the last word added is '\\end'

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

            if not any(add_more):
                break

            prev_word = word_idx
            hidden = hidden_t
        return captions, alpha


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

    def predict(self, image, B=None):
        a = self.encoder(image)
        hidden = self.init_state(a)
        if B != None:
            captions = self.decoder.predict_beam(a, hidden, B)
        else:
            captions, _ = self.decoder.predict(a, hidden)

        return captions

    def predict_with_attention(self, image):
        a = self.encoder(image)
        hidden = self.init_state(a)
        captions, alpha = self.decoder.predict(a, hidden)
        return captions, alpha
