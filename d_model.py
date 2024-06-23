import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class DModel(nn.Module):
    def __init__(self, block=ResidualBlock, all_connections=[3,4,6,3]):
        super().__init__()
        self.inputs = 16
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(16),
                        nn.ReLU()) #64x64x64
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2) #64x32x32


        self.layer0 = self.makeLayer(block, 16, all_connections[0], stride = 1) #connections = 3, shape: 64x32x32
        self.layer1 = self.makeLayer(block, 32, all_connections[1], stride = 2)#connections = 4, shape: 128x16x16
        self.layer2 = self.makeLayer(block, 128, all_connections[2], stride = 2)#connections = 6, shape: 256x8x8
        self.layer3 = self.makeLayer(block, 256, all_connections[3], stride = 2)#connections = 3, shape: 512x4x4
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(256, 1)

    def makeLayer(self, block, outputs, connections, stride=1):
        downsample = None
        if stride != 1 or self.inputs != outputs:
            downsample = nn.Sequential(
                nn.Conv2d(self.inputs, outputs, kernel_size=1, stride=stride),
                nn.BatchNorm2d(outputs),
            )
        layers = []
        layers.append(block(self.inputs, outputs, stride, downsample))
        self.inputs = outputs
        for i in range(1, connections):
            layers.append(block(self.inputs, outputs))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(-1, 256)
        x = self.fc(x).flatten()
        return F.sigmoid(x)