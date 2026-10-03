"""
model_attention.py

CNN + BiLSTM encoder with an attention LSTM decoder.
The decoder predicts one character at a time and can focus on the
part of the line that contains the next conjunct or word.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHTR(nn.Module):
    def __init__(self, num_classes, sos_idx, eos_idx, hidden=256, embed=256):
        super().__init__()
        self.num_classes = num_classes
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.hidden = hidden
        enc_dim = hidden * 2

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, 2, padding=0), nn.ReLU(inplace=True),
        )
        self.encoder = nn.LSTM(
            512, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2
        )
        self.embedding = nn.Embedding(num_classes, embed, padding_idx=0)
        self.decoder = nn.LSTMCell(embed + enc_dim, enc_dim)
        self.attn_enc = nn.Linear(enc_dim, hidden)
        self.attn_dec = nn.Linear(enc_dim, hidden)
        self.attn_v = nn.Linear(hidden, 1, bias=False)
        self.out = nn.Linear(enc_dim + enc_dim, num_classes)

    def encode(self, images):
        conv = self.cnn(images)
        b, c, h, w = conv.shape
        if h != 1:
            conv = F.adaptive_avg_pool2d(conv, (1, w))
        conv = conv.squeeze(2).permute(0, 2, 1)
        encoded, _ = self.encoder(conv)
        return encoded

    def _step(self, token, state, encoded, encoded_proj):
        emb = self.embedding(token)
        dec_h, dec_c = state
        energy = self.attn_v(torch.tanh(encoded_proj + self.attn_dec(dec_h).unsqueeze(1))).squeeze(-1)
        alpha = torch.softmax(energy, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), encoded).squeeze(1)
        dec_h, dec_c = self.decoder(torch.cat([emb, context], dim=1), (dec_h, dec_c))
        logits = self.out(torch.cat([dec_h, context], dim=1))
        return logits, (dec_h, dec_c)

    def forward(self, images, targets):
        """Teacher-forced decode. targets is (B, L) starting with SOS."""
        encoded = self.encode(images)
        encoded_proj = self.attn_enc(encoded)
        batch = images.size(0)
        state = (
            encoded.new_zeros(batch, self.hidden * 2),
            encoded.new_zeros(batch, self.hidden * 2),
        )
        outputs = []
        for t in range(targets.size(1) - 1):
            logits, state = self._step(targets[:, t], state, encoded, encoded_proj)
            outputs.append(logits)
        return torch.stack(outputs, dim=1)

    @torch.no_grad()
    def greedy_decode(self, images, max_len=128):
        self.eval()
        encoded = self.encode(images)
        encoded_proj = self.attn_enc(encoded)
        batch = images.size(0)
        state = (
            encoded.new_zeros(batch, self.hidden * 2),
            encoded.new_zeros(batch, self.hidden * 2),
        )
        token = torch.full((batch,), self.sos_idx, dtype=torch.long, device=images.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=images.device)
        generated = []
        for _ in range(max_len):
            logits, state = self._step(token, state, encoded, encoded_proj)
            token = logits.argmax(dim=1)
            token = token.masked_fill(finished, 0)
            generated.append(token)
            finished = finished | token.eq(self.eos_idx)
            if bool(finished.all()):
                break
        if not generated:
            return [[] for _ in range(batch)]
        steps = torch.stack(generated, dim=1)
        results = []
        for row in steps.tolist():
            chars = []
            for idx in row:
                if idx == self.eos_idx or idx == 0:
                    if idx == self.eos_idx:
                        break
                    continue
                chars.append(idx)
            results.append(chars)
        return results
