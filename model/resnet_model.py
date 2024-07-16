## code from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torchvision

# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152', 'ResNet34P','ResNet50S','ResNet50P','ResNet101P']
#
# resnet18_dir = '/local/sda4/yqian3/RoadNets/resnet_model/resnet18-5c106cde.pth'
# resnet34_dir = '/local/sda4/yqian3/RoadNets/resnet_model/resnet34-333f7ec4.pth'
# resnet50_dir = '/local/sda4/yqian3/RoadNets/resnet_model/resnet50-19c8e357.pth'
# resnet101_dir = '/local/sda4/yqian3/RoadNets/resnet_model/resnet101-5d3b4d8f.pth'
#
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }

def conv3x3(in_channels, out_channels, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    """
    基本块类，用于构造深度神经网络中的卷积块。
    
    参数:
    - in_channels: 输入通道数
    - out_channels: 输出通道数
    - stride: 卷积步长，默认为1
    - downsample: 下采样模块，用于匹配输入输出维度，默认为None
    
    属性:
    - expansion: 扩展倍数，本类中固定为1
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)  # 第一层3x3卷积
        self.bn1 = nn.BatchNorm2d(out_channels)  # 第一层卷积后的批量归一化
        self.relu = nn.ReLU(inplace=True)  # 激活函数，采用就地激活以减少内存占用
        self.conv2 = conv3x3(out_channels, out_channels)  # 第二层3x3卷积
        self.bn2 = nn.BatchNorm2d(out_channels)  # 第二层卷积后的批量归一化
        self.downsample = downsample  # 下采样模块，用于特征图尺寸减小时与当前块连接
        self.stride = stride  # 记录卷积步长

    def forward(self, x):
        """
        前向传播函数：执行卷积神经网络的一个基本块的前向传播。

        参数:
        - x(Tensor): 输入的特征张量。

        返回值:
        - out(Tensor): 经过该基本块处理后的输出特征张量。
        """
        # 保留输入张量，用于后续的残差连接
        residual = x

        # 第一层卷积， followed by batch normalization and ReLU activation
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二层卷积， followed by batch normalization (没有ReLU激活函数)
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果存在下采样层，则对输入进行下采样，以匹配输出的尺寸
        if self.downsample is not None:
            residual = self.downsample(x)

        # 将输出与残差相加，然后通过ReLU激活函数
        out += residual
        out = self.relu(out)

        return out

class BasicBlockDe(nn.Module):
    """
    DenseNet中的基本块，用于构建网络的每一层。
    
    参数:
    - in_channels: 输入通道数
    - out_channels: 中间层的通道数
    - stride: 卷积步长，默认为1
    - downsample: 下采样层，默认为None。如果stride!=1，则使用下采样层来匹配输入输出的维度。
    
    属性:
    - expansion: 扩展倍数，用于计算输出通道数。本类中固定为1。
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlockDe, self).__init__()

        # 构建残差连接的卷积层
        self.convRes = conv3x3(in_channels,out_channels,stride)
        self.bnRes = nn.BatchNorm2d(out_channels)
        self.reluRes = nn.ReLU(inplace=True)

        # 构建主路径的两层卷积
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        前向传播函数。
        
        参数:
        - x: 输入特征数据
        
        返回:
        - out: 经过基本块处理后的输出特征数据
        """
        # 残差连接的处理
        residual = self.convRes(x)
        residual = self.bnRes(residual)
        residual = self.reluRes(residual)

        # 主路径的处理
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果需要下采样，则对输入x进行下采样，以匹配输出维度
        if self.downsample is not None:
            residual = self.downsample(x)

        # 将主路径的输出与残差相加
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    瓶颈块类，用于构造深度神经网络中的瓶颈层，是残差网络中的一个重要组成部分。

    参数:
    - inplanes: 输入通道数
    - planes: 中间层的通道数
    - stride: 卷积步长，默认为1
    - downsample: 下采样层，默认为None。如果stride不为1，则使用下采样层来匹配输入输出的维度

    属性:
    - expansion: 扩展倍数，用于扩大输出通道数
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        前向传播函数。
        
        参数:
        - x: 输入特征数据

        返回:
        - out: 经过瓶颈层处理后的输出特征数据
        """
        residual = x  # 保存输入，用于残差连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)  # 第一层卷积，激活

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)  # 第二层卷积，激活

        out = self.conv3(out)
        out = self.bn3(out)  # 第三层卷积，不激活

        if self.downsample is not None:
            residual = self.downsample(x)  # 如果需要下采样，更新残差

        out += residual  # 残差连接
        out = self.relu(out)  # 最后一个激活函数

        return out
