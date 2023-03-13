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
        x = x.unsqueeze(1)
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
    def __init__(self, L, D, hidden_size):
        super().__init__()
        self.L = L
        self.D = D

        self.mlp = nn.Sequential(
            nn.Linear(L * (D + hidden_size), 256, bias=True),
            nn.Tanh(),
            nn.Linear(256, 128, bias=True),
            nn.Tanh(),
            nn.Linear(128, L, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, a, h):
        h = h.unsqueeze(1)
        h = torch.cat([h] * self.L, dim=1)
        batch_size = a.size(0)
        alpha = self.mlp(torch.cat((a, h), dim=2).view(batch_size, -1))
        z = torch.bmm(alpha.reshape(batch_size, 1, -1), a)

        return z.squeeze(1)


class CALSTM(nn.Module):
    def __init__(self, L, D, hidden_size, embedding_size, n_layers, vocab):
        super().__init__()
        self.vocab = vocab
        self.pad_idx = vocab.word2idx['\pad']

        self.attention = AttentionBlock(L, D, hidden_size)
        self.lstm = nn.LSTM(input_size=D + embedding_size, hidden_size=hidden_size, num_layers=n_layers,
                            batch_first=True)
        self.embed = nn.Embedding(len(vocab), embedding_size)

    def forward(self, a, hidden, y):
        padding = torch.tensor([self.pad_idx] * y.size(0), device=y.get_device()).view(-1, 1)
        y = torch.cat((padding, y[:, 1:]), dim=1)
        y = self.embed(y)
        hze = None  # concatenation to be fed into final linear layer
        for i in range(y.size(1)):
            h, c = hidden
            z = self.attention(a, h[-1])
            input = torch.cat((z, y[:, i, :]), dim=1)
            output, hidden = self.lstm(input.unsqueeze(1), hidden)
            if hze is None:
                hze = torch.cat((output.squeeze(1), z, y[:, i, :]), dim=1).unsqueeze(1)
            else:
                hze = torch.cat((hze, torch.cat((output.squeeze(1), z, y[:, i, :]), dim=1).unsqueeze(1)), dim=1)
        return hze

    def predict(self, a, hidden, prev_word=None):
        if prev_word is None:
            prev_word = torch.tensor([self.pad_idx] * a.size(0), device=a.get_device()).view(-1, 1)  # initial word is '\\pad'

        prev_word = self.embed(prev_word).squeeze(1)
        h, c = hidden
        z = self.attention(a, h[-1])
        input = torch.cat((z, prev_word), dim=1)
        output, hidden = self.lstm(input.unsqueeze(1), hidden)
        hze = torch.cat((output.squeeze(1), z, prev_word), dim=1).unsqueeze(1)
        return hze, hidden


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
        captions = []
        for _ in range(a.size(0)):
            captions.append([])
        add_more = [True] * a.size(0)  # tells model to add another word. Marked to false when the last word added is '\\end'

        for _ in range(self.max_length):
            hze, hidden = self.calstm.predict(a, hidden, prev_word=prev_word)
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
        return captions


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

    def predict(self, image):
        a = self.encoder(image)
        hidden = self.init_state(a)
        captions = self.decoder.predict(a, hidden)

        return captions

