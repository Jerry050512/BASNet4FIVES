import os
import re
import csv
from pprint import pprint
from pyeval.main import evaluate
import numpy as np

models = {
    'BASNet-PVT': [[f'BASNet-PVT-ssp-{i}' for i in range(1, 321)]]
}

method = 'python'

datasets = ['SSP2000']

results = []
best_results = []
cnt = 0


for model_name, models in models.items():
    for model in models:
        for dataset in datasets:
            cnt += 1
            if method == 'python':
                best_score = 0
                best = [None] * 5
                if isinstance(model, str):
                    model = [model]
                for model_code in model:
                    res = evaluate(model_name, dataset, model_code)
                    match = re.match(r'\[cost:(\S+)s\](\S+) dataset with (\S+) method get (\S+) mae, (\S+) S-measure..', res)
                    if match:
                        model_name_local = match.group(3)
                        Smeasure = float(match.group(5))
                        MAE = float(match.group(4))
                        maxEm = np.nan
                        maxFm = np.nan
                        res = [model_name_local, Smeasure, MAE, maxEm, maxFm]
                        if 1 - MAE + Smeasure > best_score:
                            best = res
                        results.append(res)
                best_results.append(','.join(list(map(str, best))) + '\r\n')
            else:
                raise ValueError("Unknown Method")

pprint(results)
pprint(best_results)

# 写入到CSV文件
with open('output.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # 写入表头
    csvwriter.writerow(["Model", "Smeasure", "MAE", "Max-E", "Max-F"])
    
    # 写入模型数据
    csvwriter.writerows(results)

with open('best.csv', 'w') as csvfile:
    csvfile.write("Model, Smeasure, MAE, Max-E, Max-F")
    csvfile.writelines(best_results)