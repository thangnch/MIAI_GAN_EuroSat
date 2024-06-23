import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class GModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = DoubleConv(3, 32)  # 64x64x64
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x32x32

        self.conv_2 = DoubleConv(32, 64)  # 128x32x32
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128x16x16

        self.conv_3 = DoubleConv(64, 128)  # 256x16x16
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256x8x8

        self.conv_4 = DoubleConv(128, 256)  # 512x8x8
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 512x4x4

        self.conv_5 = DoubleConv(256, 512)  # 1024x2x2

        # DECODER
        self.upconv_1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 512x4x4
        self.conv_6 = DoubleConv(512, 256)  # 512x4x4

        self.upconv_2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 256x8x8
        self.conv_7 = DoubleConv(256, 128)  # 256x8x8

        self.upconv_3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 128x16x16
        self.conv_8 = DoubleConv(128, 64)  # 128x16x16

        self.upconv_4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # 64x32x32
        self.conv_9 = DoubleConv(64, 32)  # 64x32x32

        self.output = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)  # 3x64x64

    def forward(self, batch):
        conv_1_out = self.conv_1(batch)
        conv_2_out = self.conv_2(self.pool_1(conv_1_out))
        conv_3_out = self.conv_3(self.pool_2(conv_2_out))
        conv_4_out = self.conv_4(self.pool_3(conv_3_out))
        conv_5_out = self.conv_5(self.pool_4(conv_4_out))

        conv_6_out = self.conv_6(torch.cat([self.upconv_1(conv_5_out), conv_4_out], dim=1))
        conv_7_out = self.conv_7(torch.cat([self.upconv_2(conv_6_out), conv_3_out], dim=1))
        conv_8_out = self.conv_8(torch.cat([self.upconv_3(conv_7_out), conv_2_out], dim=1))
        conv_9_out = self.conv_9(torch.cat([self.upconv_4(conv_8_out), conv_1_out], dim=1))

        output = self.output(conv_9_out)

        return F.sigmoid(output)
