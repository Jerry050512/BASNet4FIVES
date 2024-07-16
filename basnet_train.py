import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
from os.path import join, sep, exists

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import BASNet

import pytorch_ssim
import pytorch_iou

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(reduction='mean')
ssim_loss = pytorch_ssim.SSIM(window_size=11,reduction='mean')
iou_loss = pytorch_iou.IOU(reduction='mean')

def bce_ssim_loss(pred, target):
    """
    计算结合了二进制交叉熵（BCE）、结构相似度指数（SSIM）和IoU（Intersection over Union）的综合损失函数。

    参数:
    pred - 神经网络的预测输出，形状与target相同。
    target - 真实标签，形状与pred相同。

    返回值:
    loss - 综合损失值，为BCE损失、SSIM损失和IoU损失的加权之和。
    """

    # 计算二进制交叉熵损失
    bce_out = bce_loss(pred, target)
    # 计算结构相似度指数损失
    ssim_out = 1 - ssim_loss(pred, target)
    # 计算IoU损失
    iou_out = iou_loss(pred, target)

    # 综合三种损失，得到最终的损失值
    loss = bce_out + ssim_out + iou_out

    return loss

# TO-DO 重构参数顺序，更加优雅得处理多参数。
def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels_v):
    """
    对多个预测结果进行二分类交叉熵损失（BCE）融合的函数。
    
    参数:
    - d0, d1, d2, d3, d4, d5, d6, d7: 8个预测tensor，分别代表不同的模型或特征的预测结果。
    - labels_v: 真实标签的tensor。
    
    返回:
    - loss0: 第一个预测结果的BCE损失。
    - loss: 所有预测结果的BCE损失之和。
    """
    
    # 计算每个预测结果的BCE损失
    loss0 = bce_ssim_loss(d0, labels_v)
    loss1 = bce_ssim_loss(d1, labels_v)
    loss2 = bce_ssim_loss(d2, labels_v)
    loss3 = bce_ssim_loss(d3, labels_v)
    loss4 = bce_ssim_loss(d4, labels_v)
    loss5 = bce_ssim_loss(d5, labels_v)
    loss6 = bce_ssim_loss(d6, labels_v)
    loss7 = bce_ssim_loss(d7, labels_v)
    
    # 打印各个损失的值
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.item(),loss1.item(),loss2.item(),loss3.item(),loss4.item(),loss5.item(),loss6.item()))

    # 计算所有预测结果的BCE损失之和
    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7
    
    return loss0, loss


# ------- 2. set the directory of training dataset --------

data_dir = join('.', 'FIVES-dataset')
tra_image_dir = join('train', 'Ground truth')
tra_label_dir = join('train', 'Original')

image_ext = '.png'
label_ext = '.png'

model_dir = join('.', 'saved_models', 'basnet_bsi/')


epoch_num = 100000
batch_size_train = 8 
batch_size_val = 1
train_num = 0 
val_num = 0

tra_img_path_list = glob.glob(join(data_dir, tra_image_dir, '*' + image_ext))

# 生成训练标签文件名列表
# 列表中每个元素是对应训练图片的标签文件的完整路径
tra_lbl_path_list = []
for img_path in tra_img_path_list:  # 遍历训练图片路径列表
    img_name = img_path.split(sep)[-1]  # 从图片路径中提取文件名

    # 构造保存图像的文件名
    file_name_no_ext = '.'.join(img_name.split(".")[:-1])

    # 将标签文件的路径拼接成完整路径，并添加到标签文件名列表中，并去除无效的文件
    tra_lbl_path = join(data_dir, tra_label_dir, file_name_no_ext + label_ext)
    if exists(tra_lbl_path):
        tra_lbl_path_list.append(tra_lbl_path)
    else:
        tra_img_path_list.remove(img_path)

print("---")
print("train images: ", len(tra_img_path_list))
print("train labels: ", len(tra_lbl_path_list))
print("---")


train_num = len(tra_img_path_list)

# Salient Object Detection Dataset
# 显著性对象检测数据集
salobj_dataset = SalObjDataset(
    img_name_list=tra_img_path_list,
    lbl_name_list=tra_lbl_path_list,
    transform=transforms.Compose([
        RescaleT(256),
        RandomCrop(224),
        ToTensorLab(flag=0)
    ])
)
salobj_dataloader = DataLoader(
    salobj_dataset, 
    batch_size=batch_size_train, 
    shuffle=True, 
    num_workers=1
)

# ------- 3. define model --------
# define the net
net = BASNet(3, 1)
if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
# 初始化Adam优化器
#
# 参数:
# net.parameters(): 指定需要优化的网络参数
# lr=0.001: 学习率，默认为0.001，控制参数更新的速度
# betas=(0.9, 0.999): 用于计算动量的两个beta值，分别用于一阶和二阶矩的估计
# eps=1e-08: 用于避免除以零的小值添加到分母上
# weight_decay=0: 权重衰减（L2正则化）的系数，用于防止过拟合，默认为0表示不使用L2正则化
optimizer = optim.Adam(
    net.parameters(), 
    lr=0.001, 
    betas=(0.9, 0.999), 
    eps=1e-08, 
    weight_decay=0
)

# ------- 5. training process --------
print("---start training...")
net.load_state_dict(torch.load(join('.', 'saved_models', 'basnet_bsi', 'basnet.pth')))
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
avg_loss_batch_counter = 0
if __name__ == "__main__":
    for epoch in range(0, epoch_num):
        net.train()

        for i, data in enumerate(salobj_dataloader):
            ite_num += 1
            avg_loss_batch_counter += 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v = Variable(inputs.cuda(), requires_grad=False)
                labels_v = Variable(labels.cuda(), requires_grad=False)
            else:
                inputs_v = Variable(inputs, requires_grad=False)
                labels_v = Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6, d7 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.item()
            running_tar_loss += loss2.item()

            # del temporary outputs and loss
            # del d0, d1, d2, d3, d4, d5, d6, d7, loss2, loss

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / avg_loss_batch_counter, running_tar_loss / avg_loss_batch_counter))

            if ite_num % 2000 == 0:  # save model every 2000 iterations

                torch.save(net.state_dict(), model_dir + "basnet_bsi_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / avg_loss_batch_counter, running_tar_loss / avg_loss_batch_counter))
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                avg_loss_batch_counter = 0

    print('-------------Congratulations! Training Done!!!-------------')