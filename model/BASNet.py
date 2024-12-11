import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from .resnet_model import *


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
        self.conv1 = nn.Conv2d(inc_ch,32,3,padding=1)
        self.bn1 = nn.BatchNorm2d(32)
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
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.projection_dim = embed_dim // num_heads
        
        # Define the parameter matrices for the query, key, and value transformations
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Final linear layer after concatenation of the heads
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 确保query、key、value的形状是 (batch_size, seq_len, embed_dim)
        query = query.view(batch_size, -1, self.embed_dim)
        key = key.view(batch_size, -1, self.embed_dim)
        value = value.view(batch_size, -1, self.embed_dim)
        
        # 分割成多个头
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.projection_dim).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.projection_dim).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.projection_dim).transpose(1, 2)
        
        # Attention mechanism
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.projection_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Weighted sum of values based on the attention weights
        context = torch.matmul(attention_weights, value)
        
        # Concatenate the heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(context)
        
        return output, attention_weights

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        
    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, src_mask)[0]
        src = src + self.norm1(src2)
        src2 = self.feed_forward(src)
        src = src + self.norm2(src2)
        return src


class BASNet(nn.Module):
    """
    BASNet (Boundary Awareness Segmentation Network) 类。
    
    参数:
    - n_channels: 输入图像的通道数。
    - n_classes: 分割类别数量。
    """

    def __init__(self, n_channels, n_classes):
        super(BASNet,self).__init__()

        # 使用预训练的ResNet34作为编码器的基础
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        ## -------------Encoder--------------

        # 输入卷积层
        self.inconv = nn.Conv2d(n_channels, 64, 3, padding=1)
        self.inbn = nn.BatchNorm2d(64)
        self.inrelu = nn.ReLU(inplace=True)

        # ResNet的前四层（stage 1-4）
        self.encoder1 = resnet.layer1 # 224
        self.encoder2 = resnet.layer2 # 112
        self.encoder3 = resnet.layer3 # 56
        self.encoder4 = resnet.layer4 # 28
        
        # Transformer编码器层
        self.transformer_encoder = TransformerEncoderLayer(512, 4)

        # 最大池化层
        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # 自定义的第五层（stage 5）
        self.resb5_1 = BasicBlock(512, 512)
        self.resb5_2 = BasicBlock(512, 512)
        self.resb5_3 = BasicBlock(512, 512) # 14

        # 最大池化层
        self.pool5 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # 自定义的第六层（stage 6）
        self.resb6_1 = BasicBlock(512, 512)
        self.resb6_2 = BasicBlock(512, 512)
        self.resb6_3 = BasicBlock(512, 512) # 7

        ## -------------Bridge--------------

        # Bridge阶段的卷积层
        self.convbg_1 = nn.Conv2d(512, 512, 3, dilation=2, padding=2) # 7
        self.bnbg_1 = nn.BatchNorm2d(512)
        self.relubg_1 = nn.ReLU(inplace=True)
        self.convbg_m = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bnbg_m = nn.BatchNorm2d(512)
        self.relubg_m = nn.ReLU(inplace=True)
        self.convbg_2 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bnbg_2 = nn.BatchNorm2d(512)
        self.relubg_2 = nn.ReLU(inplace=True)

        ## -------------Decoder--------------

        # stage 6d 解码层
        self.conv6d_1 = nn.Conv2d(1024, 512, 3, padding=1) # 16
        self.bn6d_1 = nn.BatchNorm2d(512)
        self.relu6d_1 = nn.ReLU(inplace=True)

        self.conv6d_m = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bn6d_m = nn.BatchNorm2d(512)
        self.relu6d_m = nn.ReLU(inplace=True)

        self.conv6d_2 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bn6d_2 = nn.BatchNorm2d(512)
        self.relu6d_2 = nn.ReLU(inplace=True)

        # stage 5d 解码层
        self.conv5d_1 = nn.Conv2d(1024, 512, 3,padding=1) # 16
        self.bn5d_1 = nn.BatchNorm2d(512)
        self.relu5d_1 = nn.ReLU(inplace=True)

        self.conv5d_m = nn.Conv2d(512, 512, 3,padding=1)
        self.bn5d_m = nn.BatchNorm2d(512)
        self.relu5d_m = nn.ReLU(inplace=True)

        self.conv5d_2 = nn.Conv2d(512, 512, 3,padding=1)
        self.bn5d_2 = nn.BatchNorm2d(512)
        self.relu5d_2 = nn.ReLU(inplace=True)

        # stage 4d 解码层
        self.conv4d_1 = nn.Conv2d(1024, 512, 3,padding=1) # 32
        self.bn4d_1 = nn.BatchNorm2d(512)
        self.relu4d_1 = nn.ReLU(inplace=True)

        self.conv4d_m = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4d_m = nn.BatchNorm2d(512)
        self.relu4d_m = nn.ReLU(inplace=True)

        self.conv4d_2 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn4d_2 = nn.BatchNorm2d(256)
        self.relu4d_2 = nn.ReLU(inplace=True)

        # stage 3d 解码层
        self.conv3d_1 = nn.Conv2d(512, 256, 3, padding=1) # 64
        self.bn3d_1 = nn.BatchNorm2d(256)
        self.relu3d_1 = nn.ReLU(inplace=True)

        self.conv3d_m = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3d_m = nn.BatchNorm2d(256)
        self.relu3d_m = nn.ReLU(inplace=True)

        self.conv3d_2 = nn.Conv2d(256, 128, 3, padding=1)
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
        self.upscore6 = nn.Upsample(scale_factor=32, mode='bilinear')###
        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        ## -------------Side Output--------------
        self.outconvb = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv6 = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv5 = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv4 = nn.Conv2d(256, 1, 3, padding=1)
        self.outconv3 = nn.Conv2d(128, 1, 3, padding=1)
        self.outconv2 = nn.Conv2d(64, 1, 3, padding=1)
        self.outconv1 = nn.Conv2d(64, 1, 3, padding=1)

        ## -------------Refine Module-------------
        self.refunet = RefUnet(1, 64)


    def forward(self,x):

        hx = x

        ## -------------Encoder-------------
        hx = self.inconv(hx)
        hx = self.inbn(hx)
        hx = self.inrelu(hx)

        
        hx = self.transformer_encoder(hx)
        h1 = self.encoder1(hx) # 256
        h2 = self.encoder2(h1) # 128
        h3 = self.encoder3(h2) # 64
        h4 = self.encoder4(h3) # 32

        hx = self.pool4(h4) # 16

        hx = self.resb5_1(hx)
        hx = self.resb5_2(hx)
        h5 = self.resb5_3(hx)

        hx = self.pool5(h5) # 8

        hx = self.resb6_1(hx)
        hx = self.resb6_2(hx)
        h6 = self.resb6_3(hx)

        ## -------------Bridge-------------
        hx = self.relubg_1(self.bnbg_1(self.convbg_1(h6))) # 8
        hx = self.relubg_m(self.bnbg_m(self.convbg_m(hx)))
        hbg = self.relubg_2(self.bnbg_2(self.convbg_2(hx)))

        ## -------------Decoder-------------

        hx = self.relu6d_1(self.bn6d_1(self.conv6d_1(torch.cat((hbg,h6),1))))
        hx = self.relu6d_m(self.bn6d_m(self.conv6d_m(hx)))
        hd6 = self.relu6d_2(self.bn6d_2(self.conv6d_2(hx)))

        hx = self.upscore2(hd6) # 8 -> 16

        hx = self.relu5d_1(self.bn5d_1(self.conv5d_1(torch.cat((hx,h5),1))))
        hx = self.relu5d_m(self.bn5d_m(self.conv5d_m(hx)))
        hd5 = self.relu5d_2(self.bn5d_2(self.conv5d_2(hx)))

        hx = self.upscore2(hd5) # 16 -> 32

        hx = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((hx,h4),1))))
        hx = self.relu4d_m(self.bn4d_m(self.conv4d_m(hx)))
        hd4 = self.relu4d_2(self.bn4d_2(self.conv4d_2(hx)))

        hx = self.upscore2(hd4) # 32 -> 64

        hx = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((hx,h3),1))))
        hx = self.relu3d_m(self.bn3d_m(self.conv3d_m(hx)))
        hd3 = self.relu3d_2(self.bn3d_2(self.conv3d_2(hx)))

        hx = self.upscore2(hd3) # 64 -> 128

        hx = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((hx,h2),1))))
        hx = self.relu2d_m(self.bn2d_m(self.conv2d_m(hx)))
        hd2 = self.relu2d_2(self.bn2d_2(self.conv2d_2(hx)))

        hx = self.upscore2(hd2) # 128 -> 256

        hx = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((hx,h1),1))))
        hx = self.relu1d_m(self.bn1d_m(self.conv1d_m(hx)))
        hd1 = self.relu1d_2(self.bn1d_2(self.conv1d_2(hx)))

        ## -------------Side Output-------------
        db = self.outconvb(hbg)
        db = self.upscore6(db) # 8->256

        d6 = self.outconv6(hd6)
        d6 = self.upscore6(d6) # 8->256

        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5) # 16->256

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4) # 32->256

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3) # 64->256

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2) # 128->256

        d1 = self.outconv1(hd1) # 256

        ## -------------Refine Module-------------
        dout = self.refunet(d1) # 256

        return (
            F.sigmoid(dout), 
            F.sigmoid(d1), 
            F.sigmoid(d2), 
            F.sigmoid(d3), 
            F.sigmoid(d4), 
            F.sigmoid(d5), 
            F.sigmoid(d6), 
            F.sigmoid(db)
        )
