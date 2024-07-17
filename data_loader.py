# data loader
import warnings
import glob
import torch
from skimage import io, transform, color
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
#==========================dataset load==========================

class RescaleT(object):
    """
    用于调整数据集中的图像大小，以确保它们具有相同的输出尺寸。
    
    参数:
    - output_size: int 或 tuple。指定输出图像的大小。如果为 int，则输出图像的宽和高都将是这个值。如果为 tuple，则 output_size 应该是 (height, width) 的形式。
    
    方法:
    - __call__: 对输入的样本（包含图像和标签）进行处理，调整图像大小并返回处理后的样本。
    """

    def __init__(self, output_size):
        """
        初始化 RescaleT 类的实例。
        
        参数:
        - output_size: int 或 tuple。指定输出图像的大小。
        """
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        """
        对输入样本进行处理，调整图像大小。
        
        参数:
        - sample: 字典，包含'image'和'label'两个键。'image'是待处理的图像，'label'是对应的标签。
        
        返回:
        - 处理后的样本，字典格式，包含调整大小后的图像和标签。
        """
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]  # 获取输入图像的当前高度和宽度

        # 根据 output_size 类型计算新的图像大小
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)  # 转换为整数

        # 调整图像和标签的大小，并确保标签在调整大小时保持为整数类型
        img = transform.resize(image, (self.output_size, self.output_size), mode='constant')
        lbl = transform.resize(label, (self.output_size, self.output_size), mode='constant', order=0, preserve_range=True)

        return {'image': img, 'label': lbl}

class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'],sample['label']

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size*h/w,self.output_size
            else:
                new_h, new_w = self.output_size,self.output_size*w/h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
        img = transform.resize(image, (new_h, new_w), mode='constant')
        lbl = transform.resize(label, (new_h, new_w), mode='constant', order=0, preserve_range=True)

        return {'image':img, 'label':lbl}

class CenterCrop(object):
    """
    用于对图像进行中心裁剪的类。
    
    参数:
    - output_size: 裁剪后图像的大小，可以是整数（正方形）或元组（长宽）。
    
    方法:
    - __init__(self, output_size): 初始化函数，设置裁剪大小。
    - __call__(self, sample): 调用函数，对输入样本进行中心裁剪。
    
    返回值:
    - 裁剪后的图像和标签字典。
    """

    def __init__(self, output_size):
        """
        初始化函数。
        
        参数:
        - output_size: 裁剪后图像的大小，可以是整数（正方形）或元组（长宽）。
        """
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        """
        对输入样本进行中心裁剪。
        
        参数:
        - sample: 包含原始图像和标签的字典。
        
        返回值:
        - 裁剪后的图像和标签字典。
        """
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]  # 获取原始图像的高度和宽度
        new_h, new_w = self.output_size

        # 确保裁剪尺寸不超过原图尺寸
        assert((h >= new_h) and (w >= new_w))

        # 计算裁剪的起始位置
        h_offset = int(math.floor((h - new_h) / 2))
        w_offset = int(math.floor((w - new_w) / 2))

        # 执行裁剪
        image = image[h_offset: h_offset + new_h, w_offset: w_offset + new_w]
        label = label[h_offset: h_offset + new_h, w_offset: w_offset + new_w]

        return {'image': image, 'label': label}

class RandomCrop(object):
    """
    对样本进行随机裁剪以得到固定大小的输出。

    参数:
    - output_size: 裁剪后输出的大小，可以是整数（表示高和宽都为该值）或元组（表示高和宽的具体值）。

    方法:
    - __init__(self, output_size): 构造函数，初始化裁剪的输出大小。
    - __call__(self, sample): 将对象实例化为可调用对象，对输入样本进行裁剪处理。

    返回值:
    - 裁剪后的图像和标签字典。
    """

    def __init__(self, output_size):
        # 确保output_size是整数或长度为2的元组
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        # 从样本中提取图像和标签
        image, label = sample['image'], sample['label']

        # 获取图像当前的尺寸
        h, w = image.shape[:2]
        # 获取目标裁剪尺寸
        new_h, new_w = self.output_size

        # 随机确定裁剪的顶部和左侧位置
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        # 执行裁剪操作
        image = image[top: top + new_h, left: left + new_w]
        label = label[top: top + new_h, left: left + new_w]

        # 返回裁剪后的图像和标签
        return {'image': image, 'label': label}

class ToTensor(object):
    """转换样本中的ndarrays为Tensors。

    参数:
        sample: 一个包含'image'和'label'键的字典，其中'image'是一个ndarray，'label'是一个对应的标签ndarray。

    返回值:
        一个字典，包含经过转换后的'image'和'label'，它们被转换为torch tensors。
    """

    def __call__(self, sample):
        # 分别提取图像和标签
        image, label = sample['image'], sample['label']

        # 初始化临时图像和标签数组
        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
        tmpLbl = np.zeros(label.shape)

        # 归一化图像数据
        image = image / np.max(image)
        if(np.max(label) < 1e-6):
            label = label
        else:
            label = label / np.max(label)

        # 处理单通道图像和多通道图像，进行标准化
        if image.shape[2]==1:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        # 将标签归一化到[0,1]区间
        tmpLbl[:, :, 0] = label[:, :, 0]

        # 调整图像和标签的维度顺序以符合torch的要求
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        # 返回转换后的图像和标签
        return {'image': torch.from_numpy(tmpImg),
            'label': torch.from_numpy(tmpLbl)}

class ToTensorLab(object):
    """
    将样本中的ndarrays转换为Tensors。
    参数:
    - flag: 标志位，用于指定颜色空间的转换方式。0表示仅使用RGB颜色空间，1表示使用Lab颜色空间，2表示同时使用RGB和Lab颜色空间。
    """

    def __init__(self, flag=0):
        """
        初始化转换器。
        参数:
        - flag: 标志位，用于指定颜色空间的转换方式。默认值为0。
        """
        self.flag = flag

    def __call__(self, sample):
        """
        调用此方法将样本中的图像和标签转换为Tensor。
        参数:
        - sample: 一个字典，包含'image'和'label'两个键。'image'对应的是待转换的图像ndarray，'label'对应的是待转换的标签ndarray。
        返回:
        - 一个字典，包含转换后的'image'和'label'两个键。对应的值为转换后的图像和标签的Tensors。
        """
        image, label = sample['image'], sample['label']

        tmpLbl = np.zeros(label.shape)

        # 标签归一化
        if(np.max(label) < 1e-6):
            label = label
        else:
            label = label / np.max(label)

        # 根据flag值进行不同颜色空间的转换和归一化
        if self.flag == 2:  # 使用RGB和Lab颜色空间
            tmpImg = np.zeros((image.shape[0], image.shape[1], 6))
            tmpImgt = np.zeros((image.shape[0], image.shape[1], 3))
            # 处理单通道图像
            if image.shape[2] == 1:
                tmpImgt[:, :, 0] = image[:, :, 0]
                tmpImgt[:, :, 1] = image[:, :, 0]
                tmpImgt[:, :, 2] = image[:, :, 0]
            else:
                tmpImgt = image
            tmpImgtl = color.rgb2lab(tmpImgt)

            # 将RGB和Lab图像归一化到[0,1]
            tmpImg[:, :, 0] = (tmpImgt[:, :, 0] - np.min(tmpImgt[:, :, 0])) / (np.max(tmpImgt[:, :, 0]) - np.min(tmpImgt[:, :, 0]))
            tmpImg[:, :, 1] = (tmpImgt[:, :, 1] - np.min(tmpImgt[:, :, 1])) / (np.max(tmpImgt[:, :, 1]) - np.min(tmpImgt[:, :, 1]))
            tmpImg[:, :, 2] = (tmpImgt[:, :, 2] - np.min(tmpImgt[:, :, 2])) / (np.max(tmpImgt[:, :, 2]) - np.min(tmpImgt[:, :, 2]))
            tmpImg[:, :, 3] = (tmpImgtl[:, :, 0] - np.min(tmpImgtl[:, :, 0])) / (np.max(tmpImgtl[:, :, 0]) - np.min(tmpImgtl[:, :, 0]))
            tmpImg[:, :, 4] = (tmpImgtl[:, :, 1] - np.min(tmpImgtl[:, :, 1])) / (np.max(tmpImgtl[:, :, 1]) - np.min(tmpImgtl[:, :, 1]))
            tmpImg[:, :, 5] = (tmpImgtl[:, :, 2] - np.min(tmpImgtl[:, :, 2])) / (np.max(tmpImgtl[:, :, 2]) - np.min(tmpImgtl[:, :, 2]))

            # 标准化图像数据
            for channel in range(6):
                tmpImg[:, :, channel] = (tmpImg[:, :, channel] - np.mean(tmpImg[:, :, channel])) / np.std(tmpImg[:, :, channel])

        elif self.flag == 1:  # 使用Lab颜色空间
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

            # 处理单通道图像
            if image.shape[2] == 1:
                tmpImg[:, :, 0] = image[:, :, 0]
                tmpImg[:, :, 1] = image[:, :, 0]
                tmpImg[:, :, 2] = image[:, :, 0]
            else:
                tmpImg = image

            tmpImg = color.rgb2lab(tmpImg)

            # 将Lab图像归一化到[0,1]
            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.min(tmpImg[:, :, 0])) / (np.max(tmpImg[:, :, 0]) - np.min(tmpImg[:, :, 0]))
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.min(tmpImg[:, :, 1])) / (np.max(tmpImg[:, :, 1]) - np.min(tmpImg[:, :, 1]))
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.min(tmpImg[:, :, 2])) / (np.max(tmpImg[:, :, 2]) - np.min(tmpImg[:, :, 2]))

            # 标准化图像数据
            for channel in range(3):
                tmpImg[:, :, channel] = (tmpImg[:, :, channel] - np.mean(tmpImg[:, :, channel])) / np.std(tmpImg[:, :, channel])

        else:  # 使用RGB颜色空间
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
            # 归一化图像到[0,1]
            if np.max(image) != 0:
                image = image/np.max(image)
            else:
                warnings.warn("Image Max is 0. Can not divide by zero.")
                
            # 处理单通道图像
            if image.shape[2] == 1:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
            else:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
                tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        tmpLbl[:, :, 0] = label[:, :, 0]

        # 转换通道顺序和归一化范围
        # 将(r, g, b)转换为(b, r, g)，范围从[0, 255]转换为[0, 1]
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        return {'image': torch.from_numpy(tmpImg).float(),
                'label': torch.from_numpy(tmpLbl).float()}

class SalObjDataset(Dataset):
    def __init__(self, img_name_list, lbl_name_list, transform=None):
        """
        初始化SalObjDataset类的实例。
        
        参数:
        - img_name_list: 图像文件名列表，每个元素是图像的路径或文件名。
        - lbl_name_list: 标签文件名列表，每个元素是标签图像的路径或文件名。
        - transform: 可选的转换函数，用于对样本进行预处理。
        """
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.transform = transform

    def __len__(self):
        """
        返回数据集中的样本数量。
        
        返回:
        - int: 数据集中的样本数量。
        """
        return len(self.image_name_list)

    def __getitem__(self, idx):
        """
        根据索引获取数据集中的一个样本。
        
        参数:
        - idx: int, 要获取的样本的索引。
        
        返回:
        - dict: 包含'image'和'label'键的字典，分别对应图像和标签。
        """
        image = io.imread(self.image_name_list[idx])

        # 读取标签图像，若不存在则创建全零数组
        if(0 == len(self.label_name_list)):
            label_3 = np.zeros(image.shape)
        else:
            label_3 = io.imread(self.label_name_list[idx])

        # 处理标签图像，根据其维度将其转换为合适的格式
        label = np.zeros(label_3.shape[0: 2])
        if(3 == len(label_3.shape)):
            label = label_3[:, :, 0]
        elif(2 == len(label_3.shape)):
            label = label_3

        # 根据图像和标签的维度，添加新的维度以保持一致性
        if(3 == len(image.shape) and 2 == len(label.shape)):
            label = label[:, :, np.newaxis]
        elif(2 == len(image.shape) and 2 == len(label.shape)):
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]

        # 创建包含图像和标签的样本字典，并应用转换函数
        sample = {'image':image, 'label':label}
        if self.transform:
            sample = self.transform(sample)

        return sample