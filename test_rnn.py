import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, pool=True):
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            padding=padding
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        padding = kernel_size // 2

        self.up = nn.ConvTranspose1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=2,
            stride=2
        )

        self.conv = nn.Conv1d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            padding=padding
        )

        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class BLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size=48, output_size=96, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """
        x: [B, C, T]
        """
        x = x.transpose(1, 2)      # [B, T, C]
        x, _ = self.lstm(x)        # [B, T, hidden * 2]
        x = self.linear(x)         # [B, T, output_size]
        x = x.transpose(1, 2)      # [B, output_size, T]
        return x


class SeismicPickNet(nn.Module):
    """
    根据图中模型结构实现：
    - 输入：三分量 waveform, [B, 3, 10240]
    - Encoder: Conv1D + MaxPool
    - Bottleneck: BLSTM
    - Decoder: DeConv1D 上采样
    - Final: BLSTM + point-wise classifier
    """

    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 4, kernel_size=11)
        self.enc2 = ConvBlock(4, 8, kernel_size=9)
        self.enc3 = ConvBlock(8, 16, kernel_size=7)
        self.enc4 = ConvBlock(16, 32, kernel_size=7)
        self.enc5 = ConvBlock(32, 64, kernel_size=5)
        self.enc6 = ConvBlock(64, 64, kernel_size=3)

        # BLSTM at bottleneck
        self.bottleneck_lstm = BLSTMBlock(
            input_size=64,
            hidden_size=48,
            output_size=96
        )

        # Decoder
        self.dec1 = DeconvBlock(96, 96, kernel_size=3)
        self.dec2 = DeconvBlock(96, 96, kernel_size=5)
        self.dec3 = DeconvBlock(96, 32, kernel_size=7)
        self.dec4 = DeconvBlock(32, 32, kernel_size=7)
        self.dec5 = DeconvBlock(32, 16, kernel_size=9)
        self.dec6 = DeconvBlock(16, 8, kernel_size=11)

        # Decoder 后的 BLSTM
        self.decoder_lstm = BLSTMBlock(
            input_size=8,
            hidden_size=48,
            output_size=96
        )

        # Final RNN layer
        self.final_lstm = nn.LSTM(
            input_size=96,
            hidden_size=48,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Linear(96, num_classes)

    def forward(self, x):
        """
        x: [B, 3, 10240]
        return:
            prob: [B, num_classes, 10240]
        """

        # normalization, 对每条 waveform 做标准化
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-6
        x = (x - mean) / std

        # Encoder
        x = self.enc1(x)   # [B, 4, 5120]
        x = self.enc2(x)   # [B, 8, 2560]
        x = self.enc3(x)   # [B, 16, 1280]
        x = self.enc4(x)   # [B, 32, 640]
        x = self.enc5(x)   # [B, 64, 320]
        x = self.enc6(x)   # [B, 64, 160]

        # Bottleneck BLSTM
        x = self.bottleneck_lstm(x)  # [B, 96, 160]

        # Decoder
        x = self.dec1(x)   # [B, 96, 320]
        x = self.dec2(x)   # [B, 96, 640]
        x = self.dec3(x)   # [B, 32, 1280]
        x = self.dec4(x)   # [B, 32, 2560]
        x = self.dec5(x)   # [B, 16, 5120]
        x = self.dec6(x)   # [B, 8, 10240]

        # Decoder BLSTM
        x = self.decoder_lstm(x)     # [B, 96, 10240]

        # Final BLSTM
        x = x.transpose(1, 2)        # [B, 10240, 96]
        x, _ = self.final_lstm(x)    # [B, 10240, 96]

        logits = self.classifier(x)  # [B, 10240, num_classes]
        logits = logits.transpose(1, 2)

        prob = F.softmax(logits, dim=1)

        return prob


if __name__ == "__main__":
    model = SeismicPickNet(in_channels=3, num_classes=3)

    x = torch.randn(2, 3, 10240)
    y = model(x)

    print(y.shape)
    # torch.Size([2, 3, 10240])
