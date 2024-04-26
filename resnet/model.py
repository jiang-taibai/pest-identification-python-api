import torch.nn as nn
import torch


# ResNet34的残差结构，既包含了实线的残差又包含虚线的残差
class BasicBlock(nn.Module):
    expansion = 1  # 在残差结构中，用于主分支的卷积个数是否变化

    # 输入特征矩阵的深度 输出特征矩阵的深度 步距（1不变，2变一半） 下采样参数（对应捷径中的下采样层）
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 卷积层1
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        # bn层1
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        # 卷积层2
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        # bn层2
        self.bn2 = nn.BatchNorm2d(out_channel)
        # 下采样方法
        self.downsample = downsample

    # 定义正向传播过程
    def forward(self, x):
        identity = x
        if self.downsample is not None:  # 进行下采样
            identity = self.downsample(x)  # short cut

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        # 获得残差结构的最终输出
        return out


# ResNet50/101/152的残差结构
class Bottleneck(nn.Module):
    expansion = 4  # 残差结构中第三层卷积核个数是第一/二层卷积核个数的4倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()
        width = int(out_channel * (width_per_group / 64.)) * groups
        # 卷积层1
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)
        # BN层1
        self.bn1 = nn.BatchNorm2d(width)
        # 卷积层2
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        # BN层2
        self.bn2 = nn.BatchNorm2d(width)
        # 卷积层3
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        # BN层3
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    # 定义正确传播过程
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)  # 捷径分支 short cut

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


# 定义整个网络的框架部分
class ResNet(nn.Module):
    # block对应所定义的不同的残差结构 block_num为残差结构中conv2_x到conv5_x中残差块个数的一个列表
    # num_classes对应训练集的分类个数 include_top方便以后搭建更加复杂的网络
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True,
                 groups=1, width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.groups = groups
        self.width_per_group = width_per_group
        # 卷积层1
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7,
                               stride=2, padding=3, bias=False)
        # bn层1
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        # 最大池化下采样层
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 一系列的2， 3， 4， 5的残差结构
        self.layer1 = self._make_layer(block, 64, blocks_num[0])  # conv2_x
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)  # conv3_x
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)  # conv4_x
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)  # conv5_x
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # channel为残差结构中的第一层卷积核个数
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        # 判断是否是50/101/152的残差结构，进行调整channel
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride,
                            groups=self.groups, width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion
        # 压入实现的残差结构
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel,
                                groups=self.groups, width_per_group=self.width_per_group))
        # 转为非关键字参数
        return nn.Sequential(*layers)

    # 正向传播过程
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)
