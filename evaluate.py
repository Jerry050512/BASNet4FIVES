import torch
from skimage import measure
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

import os
from os.path import basename, exists
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob
from os.path import join, sep
from pathlib import Path

from data_loader import RescaleT
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import BASNet

def normalize_prediction(d):
    """
    标准化预测值
    参数:
    d - 输入的预测值张量
    
    返回值:
    dn - 标准化后的预测值张量
    """
    # 计算输入张量d的最大值和最小值
    ma = torch.max(d)
    mi = torch.min(d)

    # 标准化处理：将输入张量d的值映射到0-1区间
    dn = (d-mi)/(ma-mi)

    return dn

def compute_metrics(gt, pred, threshold=0.5):
    """
    计算F1-score, Dice, IoU, 和 ACC 指标。
    
    参数:
    gt - 真实标签张量
    pred - 预测张量
    threshold - 用于将概率预测转换为二进制预测的阈值
    """
    # 将预测结果转换为二进制
    binary_pred = (pred > threshold).cpu().float()

    # Flatten tensors for calculations
    gt_flat = gt.view(-1).cpu().numpy()
    pred_flat = binary_pred.view(-1).cpu().numpy()

    # Calculate metrics
    dice = measure.label(binary_pred.numpy()).mean() if len(np.unique(gt_flat)) > 1 else 0
    dice = 2 * np.sum(gt_flat * pred_flat) / (np.sum(gt_flat) + np.sum(pred_flat) + 1e-7)

    iou = np.sum(gt_flat * pred_flat) / (np.sum(np.maximum(gt_flat, pred_flat)) + 1e-7)

    # Precision and Recall for F1-score
    precision = np.sum(gt_flat * pred_flat) / (np.sum(pred_flat) + 1e-7)
    recall = np.sum(gt_flat * pred_flat) / (np.sum(gt_flat) + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    acc = accuracy_score(gt_flat, pred_flat)
    f1 = f1_score(gt_flat, pred_flat)

    return {
        'Precision': precision,
        'Recall': recall, 
        'F1-score': f1,
        'Dice': dice,
        'IoU': iou,
        'Accuracy': acc
    }

if __name__ == '__main__':
    # --------- 1. get image path and name ---------
    
    data_dir = join('.', 'FIVES-dataset', 'test')
    image_dir = 'Original'
    eval_label_dir = 'Ground Truth'
    prediction_dir = join('.', 'test_results')
    model_dir = join('.', 'saved_models', 'basnet_bsi', 'basnet_bsi_1.pth')
    
    label_ext = '.png'
    eval_img_path_list = glob.glob(join(data_dir, image_dir, '*.png'))

    eval_label_path_list = []
    for img_path in eval_img_path_list:  # 遍历训练图片路径列表
        img_name = basename(img_path)  # 从图片路径中提取文件名

        # 构造保存图像的文件名
        file_name_no_ext = '.'.join(img_name.split(".")[:-1])

        # 将标签文件的路径拼接成完整路径，并添加到标签文件名列表中，并去除无效的文件
        tra_lbl_path = join(data_dir, eval_label_dir, file_name_no_ext + label_ext)
        if exists(tra_lbl_path):
            eval_label_path_list.append(tra_lbl_path)
        else:
            eval_img_path_list.remove(img_path)
    
    # --------- 2. dataloader ---------
    #1. dataload
    test_salobj_dataset = SalObjDataset(
        img_name_list = eval_img_path_list, 
        lbl_name_list = eval_label_path_list, 
        transform=transforms.Compose([
            RescaleT(256),
            ToTensorLab(flag=0)
        ])
    )
    test_salobj_dataloader = DataLoader(
        test_salobj_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4
    )
    
    # --------- 3. model define ---------
    print("...load BASNet...")
    net = BASNet(3,1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    
    # --------- 4. inference for each image ---------
    metrics = []
    for i_test, data_test in enumerate(test_salobj_dataloader):
    
        print("inferencing:", eval_img_path_list[i_test].split(sep)[-1])
    
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
    
        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)
    
        d1, *_ = net(inputs_test)
    
        # normalization
        pred = d1[:,0,:,:]
        pred = normalize_prediction(pred)
        gt = data_test['label']
        metrics.append(compute_metrics(gt, pred))
        print(metrics[-1])

    # 计算平均指标
    avg_metrics = {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}
    print('--------------------')
    print('Average Metrics: ')
    print(avg_metrics)