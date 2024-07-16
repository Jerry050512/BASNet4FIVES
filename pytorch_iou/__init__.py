import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# TODO: 修改注释

def _iou(pred, target, reduction = 'mean'):
    """
    计算预测框和目标框的交并比（IoU）损失。

    参数:
    pred: 预测框的张量，形状为(batch_size, channels, height, width)。
    target: 目标框的张量，形状与pred相同。
    size_average: 是否按batch_size平均计算IoU损失，默认为True。

    返回值:
    IoU的平均值或总和，取决于size_average参数。
    """

    b = pred.shape[0]  # 获取batch_size
    IoU = 0.0
    for i in range(0, b):
        # 计算第i个预测框和目标框的IoU
        Iand1 = torch.sum(target[i,:,:,:] * pred[i,:,:,:])  # 交集面积
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:]) - Iand1  # 并集面积
        IoU1 = Iand1 / Ior1  # 第i个预测框和目标框的IoU

        # 计算IoU损失，累加到IoU上
        IoU = IoU + (1 - IoU1)

    # 根据size_average参数决定返回IoU的平均值或总和
    return IoU/b

class IOU(torch.nn.Module):
    """
    IOU类用于计算交并比（Intersection Over Union），继承自torch.nn.Module。

    参数:
    - size_average (bool): 是否对批处理中的所有样本的IOU求平均。默认为True。

    方法:
    - forward: 计算预测框和目标框的IOU。
    """
    def __init__(self, reduction = 'mean'):
        """
        初始化IOU类实例。

        参数:
        - size_average (bool): 是否对批处理中的所有样本的IOU求平均。默认为True。
        """
        super(IOU, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        """
        前向传播函数，计算预测框(pred)和目标框(target)的IOU。

        参数:
        - pred: 预测框的张量。
        - target: 目标框的张量。

        返回:
        - IOU的计算结果。
        """
        return _iou(pred, target, self.reduction)  # 调用_iou函数计算IOU
