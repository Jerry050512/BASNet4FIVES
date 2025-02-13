import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

# from .resnet_model import *
from .pvt import pvt_small


class RefUnet(nn.Module):
    """
    RefUnet 类：基于Reference U-Net架构的网络模型。
    
    参数:
    - in_ch: 输入通道数。
    - inc_ch: 增加的通道数。
    """
    def __init__(self,in_ch,inc_ch):
        super(RefUnet, self).__init__()

        # 初始卷积层
        self.conv0 = nn.Conv2d(in_ch,inc_ch,3,padding=1)

        # 第一层卷积模块
        self.conv1 = nn.Conv2d(inc_ch,64,3,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        # 第二层卷积模块
        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        # 第三层卷积模块
        self.conv3 = nn.Conv2d(64,64,3,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        # 第四层卷积模块
        self.conv4 = nn.Conv2d(64,64,3,padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        # 下采样部分的结束，接下来是上采样部分

        # 第五层卷积模块（上采样前）
        self.conv5 = nn.Conv2d(64,64,3,padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        # 重叠的上采样和卷积模块，用于结合下采样的特征
        self.conv_d4 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        # 最终的卷积层，用于输出
        self.conv_d0 = nn.Conv2d(64,1,3,padding=1)

        # 双线性插值上采样，用于恢复图像分辨率
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self,x):
        """
        实现前向传播过程，用于卷积神经网络的特征提取和图像重建。
        
        参数:
        - x : 输入图像的张量
        
        返回值:
        - 加权残差和输入图像的和，用于图像的最终输出。
        """
        
        # 初始输入图像保持不变
        hx = x
        # 应用第一层卷积
        hx = self.conv0(hx)

        # 应用第二层卷积，包括激活函数和批量归一化
        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        # 使用池化层降低维度
        hx = self.pool1(hx1)

        # 应用第三层卷积
        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        # 应用池化层
        hx = self.pool2(hx2)

        # 应用第四层卷积
        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        # 应用池化层
        hx = self.pool3(hx3)

        # 应用第五层卷积
        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        # 应用池化层
        hx = self.pool4(hx4)

        # 应用第六层卷积，不过不进行池化
        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        # 通过上采样层恢复分辨率
        hx = self.upscore2(hx5)

        # 进行特征融合，结合第四层的特征，应用卷积、批量归一化和激活函数
        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx, hx4), 1))))
        # 使用上采样层进一步恢复分辨率
        hx = self.upscore2(d4)

        # 同样的过程用于第三层特征的融合和分辨率恢复
        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx, hx3), 1))))
        hx = self.upscore2(d3)

        # 对第二层特征进行融合和分辨率恢复
        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx, hx2), 1))))
        hx = self.upscore2(d2)

        # 对第一层特征进行融合，不涉及上采样，直接输出残差
        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx, hx1), 1))))

        # 计算残差，用于与原始输入图像相加
        residual = self.conv_d0(d1)

        # 返回加权残差和输入图像的和
        return x + residual

class BASNet(nn.Module):
    """
    BASNet (Boundary Awareness Segmentation Network) 类。
    
    参数:
    - n_channels: 输入图像的通道数。
    - n_classes: 分割类别数量。
    """

    def __init__(self, n_channels, n_classes):
        super(BASNet,self).__init__()

        ## -------------Encoder--------------

        # 输入卷积层
        self.inconv = nn.Conv2d(n_channels, 64, 3, padding=1)
        self.inbn = nn.BatchNorm2d(64)
        self.inrelu = nn.ReLU(inplace=True)

        # 1. 替换ResNet34为PVT-Small
        self.pvt = pvt_small()  # 替换原来的ResNet34

        ## -------------Bridge--------------

        # Bridge阶段的卷积层
        self.convbg_1 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bnbg_1 = nn.BatchNorm2d(512)
        self.relubg_1 = nn.ReLU(inplace=True)
        self.convbg_m = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bnbg_m = nn.BatchNorm2d(512)
        self.relubg_m = nn.ReLU(inplace=True)
        self.convbg_2 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bnbg_2 = nn.BatchNorm2d(512)
        self.relubg_2 = nn.ReLU(inplace=True)

        ## -------------Decoder--------------

        # stage 4d 解码层
        self.conv4d_1 = nn.Conv2d(1024, 512, 3,padding=1) # 32
        self.bn4d_1 = nn.BatchNorm2d(512)
        self.relu4d_1 = nn.ReLU(inplace=True)

        self.conv4d_m = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4d_m = nn.BatchNorm2d(512)
        self.relu4d_m = nn.ReLU(inplace=True)

        self.conv4d_2 = nn.Conv2d(512, 320, 3, padding=1)
        self.bn4d_2 = nn.BatchNorm2d(320)
        self.relu4d_2 = nn.ReLU(inplace=True)

        # stage 3d 解码层
        self.conv3d_1 = nn.Conv2d(640, 320, 3, padding=1) # 64
        self.bn3d_1 = nn.BatchNorm2d(320)
        self.relu3d_1 = nn.ReLU(inplace=True)

        self.conv3d_m = nn.Conv2d(320, 320, 3, padding=1)
        self.bn3d_m = nn.BatchNorm2d(320)
        self.relu3d_m = nn.ReLU(inplace=True)

        self.conv3d_2 = nn.Conv2d(320, 128, 3, padding=1)
        self.bn3d_2 = nn.BatchNorm2d(128)
        self.relu3d_2 = nn.ReLU(inplace=True)

        # stage 2d 解码层
        self.conv2d_1 = nn.Conv2d(256, 128, 3, padding=1) # 128
        self.bn2d_1 = nn.BatchNorm2d(128)
        self.relu2d_1 = nn.ReLU(inplace=True)

        self.conv2d_m = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2d_m = nn.BatchNorm2d(128)
        self.relu2d_m = nn.ReLU(inplace=True)

        self.conv2d_2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn2d_2 = nn.BatchNorm2d(64)
        self.relu2d_2 = nn.ReLU(inplace=True)

        # stage 1d 解码层
        self.conv1d_1 = nn.Conv2d(128, 64, 3, padding=1) # 256
        self.bn1d_1 = nn.BatchNorm2d(64)
        self.relu1d_1 = nn.ReLU(inplace=True)

        self.conv1d_m = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1d_m = nn.BatchNorm2d(64)
        self.relu1d_m = nn.ReLU(inplace=True)

        self.conv1d_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1d_2 = nn.BatchNorm2d(64)
        self.relu1d_2 = nn.ReLU(inplace=True)

        ## -------------Bilinear Upsampling--------------
        # 双线性插值上采样层
        self.upscore = nn.Upsample(size=(224, 224), mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        ## -------------Side Output--------------
        self.outconvb = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv4 = nn.Conv2d(320, 1, 3, padding=1)
        self.outconv3 = nn.Conv2d(128, 1, 3, padding=1)
        self.outconv2 = nn.Conv2d(64, 1, 3, padding=1)
        self.outconv1 = nn.Conv2d(64, 1, 3, padding=1)

        ## -------------Refine Module-------------
        self.refunet = RefUnet(1, 64)


    def forward(self,x):

        hx = x
        hx = self.inrelu(self.inbn(self.inconv(x)))

        ## -------------Encoder-------------
        # 获取PVT的多尺度特征
        pvt_features = self.pvt(x)
        # [torch.Size([1, 64, 56, 56]), torch.Size([1, 128, 28, 28]), torch.Size([1, 320, 14, 14]), torch.Size([1, 512, 7, 7])]
        
        # 转换特征通道数以匹配原架构
        h1 = pvt_features[0]  # 64x56x56
        h2 = pvt_features[1]  # 128x28x28
        h3 = pvt_features[2]  # 320x14x14
        h4 = pvt_features[3]  # 512x7x7

        ## -------------Bridge-------------
        hx = self.relubg_1(self.bnbg_1(self.convbg_1(h4))) # 512x7x7
        hx = self.relubg_m(self.bnbg_m(self.convbg_m(hx)))
        hbg = self.relubg_2(self.bnbg_2(self.convbg_2(hx)))

        ## -------------Decoder-------------

        hx = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((hbg,h4),1))))
        hx = self.relu4d_m(self.bn4d_m(self.conv4d_m(hx)))
        hd4 = self.relu4d_2(self.bn4d_2(self.conv4d_2(hx))) # 320x7x7

        hx = self.upscore2(hd4) # 7 -> 14

        hx = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((hx,h3),1))))
        hx = self.relu3d_m(self.bn3d_m(self.conv3d_m(hx)))
        hd3 = self.relu3d_2(self.bn3d_2(self.conv3d_2(hx))) # 128x14x14

        hx = self.upscore2(hd3) # 14 -> 28

        hx = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((hx,h2),1))))
        hx = self.relu2d_m(self.bn2d_m(self.conv2d_m(hx)))
        hd2 = self.relu2d_2(self.bn2d_2(self.conv2d_2(hx))) # 64x28x28

        hx = self.upscore2(hd2) # 28 -> 56

        hx = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((hx,h1),1))))
        hx = self.relu1d_m(self.bn1d_m(self.conv1d_m(hx)))
        hd1 = self.relu1d_2(self.bn1d_2(self.conv1d_2(hx))) # 64x56x56

        ## -------------Side Output-------------
        db = self.outconvb(hbg)
        db = self.upscore(db) # 7

        d4 = self.outconv4(hd4)
        d4 = self.upscore(d4) # 7

        d3 = self.outconv3(hd3)
        d3 = self.upscore(d3) # 14

        d2 = self.outconv2(hd2)
        d2 = self.upscore(d2) # 28

        d1 = self.outconv1(hd1) # 56
        d1 = self.upscore(d1)

        ## -------------Refine Module-------------
        dout = self.refunet(d1)

        return (
            dout, 
            d1, 
            d2, 
            d3, 
            d4, 
            db
        )

if __name__ == '__main__':
    # Load the PVT model
    model = BASNet(3, 1)

    # # Optionally load pre-trained weights
    # pretrained_weights_path = 'path/to/pretrained.pth'
    # model.init_weights(pretrained_weights_path)

    # Input tensor (assuming batch size 1)
    input_tensor = torch.randn(1, 3, 224, 224)

    # Forward pass
    output = model(input_tensor)
    print([o.shape for o in output])