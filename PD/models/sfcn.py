import monai
import torch.nn as nn
from monai.networks.blocks import Convolution


class SFCN(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(monai.networks.blocks.Convolution(3, 1, 2, strides=1, kernel_size=3),
                                    nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3)))
        self.block2 = nn.Sequential(monai.networks.blocks.Convolution(3, 2, 4, strides=1, kernel_size=3),
                                    nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.block3 = nn.Sequential(monai.networks.blocks.Convolution(3, 4, 8, strides=1, kernel_size=3),
                                    nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.block4 = nn.Sequential(monai.networks.blocks.Convolution(3, 8, 8, strides=1, kernel_size=3),
                                    nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.block5 = nn.Sequential(monai.networks.blocks.Convolution(3, 8, 16, strides=1, kernel_size=3),
                                    nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.block6 = nn.Sequential(monai.networks.blocks.Convolution(3, 16, 32, strides=1, kernel_size=3),
                                    nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))

        self.conv1x1 = monai.networks.blocks.Convolution(3, 32, 32, strides=1, kernel_size=1)
        self.avgpool1 = nn.AvgPool3d(kernel_size=(1, 1, 1))
        self.dropout1 = nn.Dropout(.5)
        self.flat1 = nn.Flatten()

        self.block7 = nn.Sequential(self.conv1x1, self.avgpool1, self.dropout1, self.flat1)
        self.fc1 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.fc1(x)
        return x
