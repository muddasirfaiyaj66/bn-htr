"""
model.py

CRNN (CNN + BiLSTM) architecture for offline handwritten line recognition,
trained with CTC loss. Sized to comfortably fit an 8GB GPU (e.g. RTX 4060)
at batch sizes of 32-64 with images of height 64px.
"""

import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_classes, img_h=64, rnn_hidden=256):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),      # H/2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),    # H/4
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1)),  # H/8
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1)),  # H/16
            nn.Conv2d(512, 512, 2, padding=0), nn.ReLU(inplace=True),                        # collapse H to 1
        )

        # After the CNN, height should collapse to 1 (for img_h=64: 64/16=4, then kernel 2 -> 1... verify at runtime)
        self.rnn = nn.LSTM(512, rnn_hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.25)
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x):
        conv = self.cnn(x)                  # (B, C, H', W')
        b, c, h, w = conv.shape
        if h != 1:
            # safety net in case input height isn't exactly 64 -> pool remaining height
            conv = nn.functional.adaptive_avg_pool2d(conv, (1, w))
        conv = conv.squeeze(2)              # (B, C, W)
        conv = conv.permute(0, 2, 1)        # (B, W, C)  -- sequence over width
        rnn_out, _ = self.rnn(conv)         # (B, W, 2*hidden)
        logits = self.fc(rnn_out)           # (B, W, num_classes)
        return logits


if __name__ == "__main__":
    # quick shape sanity check
    model = CRNN(num_classes=120)
    dummy = torch.randn(2, 1, 64, 800)
    out = model(dummy)
    print("Output shape:", out.shape)  # expect (2, T, 120)
