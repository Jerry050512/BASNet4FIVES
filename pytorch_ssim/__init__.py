# https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    """
    生成一个高斯窗口函数。

    参数:
    window_size (int): 窗口的大小。必须为正整数。
    sigma (float): 高斯函数的标准差。必须为正数。

    返回:
    torch.Tensor: 形状为[window_size]的Tensor，包含了根据指定窗口大小和标准差生成的高斯函数值。
    """
    # 根据高斯函数公式，计算每个位置上的值，并存储在Tensor中
    gauss = torch.Tensor([
        exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) 
        for x in range(window_size)
    ])
    # 将高斯函数值按总和归一化，确保窗口内所有值的和为1
    return gauss / gauss.sum()

def create_window(window_size, channel):
    """
    创建一个指定大小的高斯窗口。

    参数:
    window_size (int): 窗口的大小（边长）。
    channel (int): 窗口的通道数。

    返回:
    window (Variable): 扩展后的高斯窗口，维度为(batch_size, channel, window_size, window_size)。
    """
    # 创建一个1D的高斯窗口，然后转换为2D窗口
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    
    # 将2D窗口扩展为指定通道数的窗口，并确保其连续性
    window = Variable(
        _2D_window.expand(
            channel, 
            1, 
            window_size, 
            window_size
        ).contiguous()
    )
    
    return window

def _ssim(img1, img2, window, window_size, channel, reduction = 'mean'):
    """
    计算两个图像的结构相似度指数(SSIM)

    参数:
    img1, img2: 输入的两个图像张量
    window: 用于计算局部区域的卷积窗口（核）
    window_size: 窗口大小，决定了计算SSIM的局部区域的大小
    channel: 图像的通道数
    size_average: 是否对整个图像的SSIM值进行平均，默认为True

    返回值:
    如果size_average为True，则返回两个图像的平均SSIM值；
    如果size_average为False，则返回每个通道的SSIM值的平均值。
    """

    # 计算两个图像的均值
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    # 计算均值的平方和均值的乘积
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 计算方差和协方差
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    # 用于稳定计算的常数
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # 计算SSIM映射
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # 根据参数，返回平均SSIM值或每个通道的SSIM值的平均值
    if reduction == 'mean':
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    """
    SSIM指标计算类，用于计算两个图像的结构相似度指数（Structural Similarity Index）。
    
    参数:
    - window_size: 窗口大小，默认为11。计算SSIM时使用的滑动窗口的大小。
    - size_average: 是否对整个图像的SSIM值进行平均，默认为True。
    """
    def __init__(self, window_size = 11, reduction = 'mean'):
        """
        初始化SSIM模块。
        """
        super(SSIM, self).__init__()
        self.window_size = window_size  # 窗口大小
        self.reduction = reduction  # 是否平均
        self.channel = 1  # 默认通道数为1
        self.window = create_window(window_size, self.channel)  # 创建窗口

    def forward(self, img1, img2):
        """
        计算两个图像的SSIM值。
        
        参数:
        - img1: 输入图像1。
        - img2: 输入图像2。
        
        返回:
        - 计算得到的SSIM值。
        """
        (_, channel, _, _) = img1.size()  # 获取输入图像的尺寸

        # 如果通道数和窗口数据类型与当前设置一致，则重用先前创建的窗口
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            # 否则，根据输入图像的通道数创建新的窗口
            window = create_window(self.window_size, channel)

            # 如果是在GPU上运行，则将窗口数据移动到对应的设备上，并确保其数据类型与输入图像一致
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            # 更新模块的窗口和通道数属性
            self.window = window
            self.channel = channel

        # 计算并返回SSIM值
        return _ssim(img1, img2, window, self.window_size, channel, self.reduction)

def _logssim(img1, img2, window, window_size, channel, reduction = 'mean'):
    """
    计算两个图像的Log SSIM（结构相似度指数）值。

    参数:
    - img1, img2: 输入的两个图像张量。
    - window: 用于计算局部均值和方差的卷积窗口（核）。
    - window_size: 窗口的大小，决定了计算SSIM的局部区域大小。
    - channel: 图像的通道数。
    - size_average: 是否对整个图像的SSIM值进行平均。默认为True。

    返回值:
    - 如果size_average为True，则返回两个图像的平均Log SSIM值；
    - 如果size_average为False，则返回每个通道的Log SSIM值的平均值。
    """

    # 计算两个图像的局部均值。
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    # 计算均值的平方和平方的均值。
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 计算图像的局部方差和协方差。
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    # SSIM的平滑常数。
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # 计算SSIM指数。
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # 标准化SSIM指数，使其在0到1之间。
    ssim_map = (ssim_map - torch.min(ssim_map)) / (torch.max(ssim_map) - torch.min(ssim_map))
    
    # 计算Log SSIM。
    ssim_map = -torch.log(ssim_map + 1e-8)

    # 根据参数，返回整个图像的平均Log SSIM值或每个通道的平均Log SSIM值。
    if reduction == 'mean':
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class LOGSSIM(torch.nn.Module):
    """
    LOGSSIM (Logarithmic Structural Similarity Index) 模块，用于计算两个图像的LOGSSIM值。
    
    参数:
    - window_size: int, 窗口大小，默认为11。
    - size_average: bool, 是否对结果进行平均，默认为True。
    """
    def __init__(self, window_size = 11, reduction = 'mean'):
        """
        初始化LOGSSIM模块。
        """
        super(LOGSSIM, self).__init__()
        self.window_size = window_size  # 窗口大小
        self.reduction = reduction  # 是否平均
        self.channel = 1  # 默认通道数
        self.window = create_window(window_size, self.channel)  # 创建窗口

    def forward(self, img1, img2):
        """
        前向传播计算LOGSSIM。
        
        参数:
        - img1: 图像1
        - img2: 图像2
        
        返回:
        - LOGSSIM值
        """
        (_, channel, _, _) = img1.size()  # 获取输入图像的尺寸

        # 如果通道数匹配且窗口数据类型与输入图像相同，则使用预创建的窗口
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            # 否则，根据输入图像的通道数创建新的窗口
            window = create_window(self.window_size, channel)

            # 如果输入图像在GPU上，则将窗口移动到相应的设备，并确保窗口数据类型与输入图像相同
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window  # 更新模块的窗口
            self.channel = channel  # 更新模块的通道数

        # 计算LOGSSIM值
        return _logssim(img1, img2, window, self.window_size, channel, self.reduction)


def ssim(img1, img2, window_size = 11, reduction = True):
    """
    计算两个图像的结构相似性指数(SSIM)。
    
    参数:
    - img1: 图像1
    - img2: 图像2
    - window_size: int, 窗口大小，默认为11。
    - size_average: bool, 是否对结果进行平均，默认为True。
    
    返回:
    - SSIM值
    """
    (_, channel, _, _) = img1.size()  # 获取输入图像的尺寸
    window = create_window(window_size, channel)  # 创建窗口

    # 如果输入图像在GPU上，则将窗口移动到相应的设备，并确保窗口数据类型与输入图像相同
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    # 计算SSIM值
    return _ssim(img1, img2, window, window_size, channel, reduction)