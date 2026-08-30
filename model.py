"""
model.py

CRNN (CNN + BiLSTM) for offline handwritten line recognition with CTC.
"""

import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_classes, img_h=64, rnn_hidden=256):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, 2, padding=0), nn.ReLU(inplace=True),
        )

        self.rnn = nn.LSTM(
            512, rnn_hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.25
        )
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.shape
        if h != 1:
            conv = nn.functional.adaptive_avg_pool2d(conv, (1, w))
        conv = conv.squeeze(2).permute(0, 2, 1)
        rnn_out, _ = self.rnn(conv)
        return self.fc(rnn_out)


if __name__ == "__main__":
    model = CRNN(num_classes=120)
    dummy = torch.randn(2, 1, 64, 800)
    out = model(dummy)
    print("Output shape:", out.shape)
