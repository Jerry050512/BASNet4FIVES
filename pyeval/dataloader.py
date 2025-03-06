from torch.utils import data
import torch
import glob
import os
from os.path import basename, join
from PIL import Image
SAL_DIR = "/home/shenkunye/GangZheng/dataset/prediction/{model}/{dataset}/{mode}/"
class EvalDataset(data.Dataset):
    def __init__(self, model: str, dataset: str, model_code: str, mode='test'):
        self.image_path = sorted(glob.glob(f"{SAL_DIR.format(model=model, dataset=dataset, mode=mode)}*_{model_code}.png"))
        self.label_path = [f"{SAL_DIR.format(model=model, dataset=dataset, mode=mode)}{basename(image).split('_')[0]}.png" for image in self.image_path]
        # print(self.image_path)
        # print(self.image_path.sort())
        print(f"Load {len(self)} images for {model_code} at {dataset}")
        self.__model = model
        self.__dataset = dataset
        self.__model_code = model_code
        self.__mode = mode

    def clear(self):
        for image in self.image_path:
            os.remove(image)
        self.reload()
    
    def reload(self):
        self.__init__(self.__model, self.__dataset, self.__model_code, self.__mode)

    def __getitem__(self, item):
        pred = Image.open(self.image_path[item]).convert('L')
        gt = Image.open(self.label_path[item]).convert('L')
        # print(self.image_path[item], self.label_path[item])
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)
        return pred, gt

    def __len__(self):
        return len(self.image_path)