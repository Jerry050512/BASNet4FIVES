import torch
import torch.nn as nn
import argparse
import os.path as osp
import os
from .evaluator import Eval_thread
from .dataloader import EvalDataset
import re
import numpy as np

# # from concurrent.futures import ThreadPoolExecutor
# def main(cfg):
#     root_dir = cfg.root_dir
#     if cfg.save_dir is not None:
#         output_dir = cfg.save_dir
#     else:
#         output_dir = root_dir
#     gt_dir = osp.join(root_dir, 'gt')
#     pred_dir = osp.join(root_dir, 'pred')
#     if cfg.methods is None:
#         method_names = os.listdir(pred_dir)
#     else:
#         method_names = cfg.methods.split(' ')
#     if cfg.datasets is None:
#         dataset_names = os.listdir(gt_dir)
#     else:
#         dataset_names = cfg.datasets.split(' ')
    
#     threads = []
#     for method in method_names:
#         for dataset in dataset_names:
#             loader = EvalDataset(osp.join(pred_dir, method, dataset), osp.join(gt_dir, dataset))
#             thread = Eval_thread(loader, method, dataset, output_dir, cfg.cuda)
#             threads.append(thread)
#     for thread in threads:
#         print(thread.run())

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--methods', type=str, default=None)
#     parser.add_argument('--datasets', type=str, default=None)
#     parser.add_argument('--root_dir', type=str, default="./data_compare/")
#     parser.add_argument('--save_dir', type=str, default=None)
#     parser.add_argument('--cuda', type=bool, default=True)
#     config = parser.parse_args()
#     main(config)
def clear_pred(model: str, dataset: str, model_code: str, mode='test'):
    EvalDataset(model, dataset, model_code, mode).clear()

def evaluate(model: str, dataset: str, model_code: str, mode='test'):
    loader = EvalDataset(model, dataset, model_code, mode)
    thread = Eval_thread(loader, model_code, dataset, "./", True)
    res = thread.run()
    print(res)
    return res

def evaluate_mae_smeasure(model_name, dataset, model_code):
    res = evaluate(model_name, dataset, model_code)
    match = re.match(r'\[cost:(\S+)s\](\S+) dataset with (\S+) method get (\S+) mae, (\S+) S-measure..', res)
    if match:
        model_name = match.group(3)
        Smeasure = float(match.group(5))
        MAE = float(match.group(4))
        maxEm = np.nan
        maxFm = np.nan
        res = {
            "model": model_name,
            "Smeasure": Smeasure,
            "MAE": MAE,
            "score": 1 - MAE + Smeasure,
            "message": res
        }
        return res
        

if __name__ == '__main__':
    evaluate('BASNet-PVT', 'SSP2000', 'BASNet-PVT-ssp-320')