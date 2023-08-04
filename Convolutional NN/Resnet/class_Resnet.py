import torch
import torch.nn as nn
from torch.nn import functional as F


# In this .py file, we construct ResNet as a class

# create a residual block
class Residual(nn.Module):
    def __init__(self, nin, nout, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(nin, nout, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(nout, nout, kernel_size=3, padding=1)
        if use_1x1conv:  # blk的两种结构
            self.conv3 = nn.Conv2d(nin, nout, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(nout)
        self.bn2 = nn.BatchNorm2d(nout)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        if self.conv3:
            X = self.conv3(X)

        Y += X  # 加上残差
        return F.relu(Y)


# With out a bottleneck block used in ResNet50 and deeper network.
class Resnet(nn.Module):
    def __init__(self, n_classes):
        super(Resnet, self).__init__()
        self.hidden_channels = 64  # 定义隐藏层大小
        self.block1 = nn.Sequential(nn.Conv2d(3, self.hidden_channels, kernel_size=7, stride=2, padding=3),
                                    nn.BatchNorm2d(self.hidden_channels), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.block2 = self.make_resblock(self.hidden_channels, self.hidden_channels, is_first=True)
        # Due to the same channel numbers of the block2, the 1 x 1 conv will not be used to adjust the number of input and output channels.
        self.block3 = self.make_resblock(self.hidden_channels, 128)
        self.block4 = self.make_resblock(128, 256)
        self.block5 = self.make_resblock(256, 512)
        self.output_block = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                          nn.Flatten(),
                                          nn.Linear(512, n_classes))


    def make_resblock(self, nin, nout, n_residuals=2, is_first=False):
        blk = []

        for i in range(n_residuals):
            if i == 0 and not is_first:
                blk.append(Residual(nin, nout, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(nout, nout))

        return nn.Sequential(*blk)


    def forward(self, X):
        out = self.block1(X)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.output_block(out)

        return out


if __name__ == '__main__':
    test_net = Resnet(10)
    x = torch.rand(1, 3, 224, 224)
    y = test_net(x)
    print(y.size())
