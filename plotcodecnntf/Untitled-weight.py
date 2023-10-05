import re
import torch
import numpy as np 
import matplotlib.mlab as mlab 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import operator
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import math
import networkx as nx
import pandas as pd
import copy
import csv
from scipy.stats import norm
from matplotlib.collections import LineCollection
import seaborn as sns 
import matplotlib as mpl 
from torch import tensor
from matplotlib.collections import PolyCollection
torch.set_printoptions(threshold=np.inf)

def weightmean(dictionary):
    tensor1 = dictionary.flatten()
    return tensor1.mean()

input_size = 28 * 28
output_size = 10
node1 = 5
node2 = 5

def extract_weights(file_path, layer_name):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    capture = False
    epoch = None
    weight_lines = ''

    for line in lines:
        if "epoch" in line:
            epoch = int(re.search(r'\d+', line).group())
            continue

        if layer_name in line:
            capture = True
            weight_lines = ''
            continue

        if capture:
            weight_lines += line.strip()
            if weight_lines.count('[') == weight_lines.count(']'):
                weight_lines = weight_lines[7:-1]
                weights = eval(weight_lines)
                weight_tensor = torch.tensor(weights)
                capture = False
                yield epoch, weight_tensor



# fig, ax = plt.subplots(dpi=300, figsize=(10, 6))
# ax.set_ylim([0, 100]) 

model_paths = []    
for i in range(13, 22):
    for j in range(1,11):
        path = "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultcnn/b100l5lr0001-{}/train_result{}/weights.txt".format(i, j)
        model_paths.append(path)
for i in range(7, 8):
    for j in range(1,11):
        path = "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultcnn/b100l5lr0001-{}/train_result{}/weights.txt".format(i, j)
        model_paths.append(path)
        
model_paths1 = []
model_paths2 = []
model_paths3 = []
model_paths4 = []
model_paths5 = []
test_acc_diff1 = []
test_acc_diff2 = []
test_acc_diff3 = []
# Initialize variables to keep track of the closest and maximum test accuracy
closest_acc = float('inf')
max_acc = float('-inf')

# Loop through all model paths
for path in model_paths:
    history_path = path.replace('weights.txt', 'train_history_dic.pkl')
    with open(history_path, 'rb') as g:
        history = torch.load(g)
    # Get test accuracy from the history dictionary
    test_acc = history['model_epoch20']['test_acc']
        # Check if the test accuracy is less than 0.2
    if test_acc < 0.2:
        model_paths1.append(path)
    # Check if the test accuracy is in the range [0.2, 0.5)
    if 0.2 <= test_acc < 0.5:
         model_paths2.append(path)
    # compute the difference between test_acc and 0.65
    diff1 = abs(test_acc - 0.7)
    # append the path and the difference to the corresponding lists
    model_paths3.append(path)
    test_acc_diff1.append(diff1)

    diff2 = abs(test_acc - 1)
    # append the path and the difference to the corresponding lists
    model_paths4.append(path)
    test_acc_diff2.append(diff2)
        
    diff3 = abs(test_acc - 0)
    # append the path and the difference to the corresponding lists
    model_paths5.append(path)
    test_acc_diff3.append(diff3)
# get the indices of the 100 smallest differences
indices1 = sorted(range(len(test_acc_diff1)), key=lambda i: test_acc_diff1[i])[:10]
# use the indices to get the corresponding paths
model_paths3 = [model_paths3[i] for i in indices1]
# get the indices of the 100 smallest differences
indices2 = sorted(range(len(test_acc_diff2)), key=lambda i: test_acc_diff2[i])[:10]
# use the indices to get the corresponding paths
model_paths4 = [model_paths4[i] for i in indices2]
# get the indices of the 100 smallest differences
indices3 = sorted(range(len(test_acc_diff3)), key=lambda i: test_acc_diff3[i])[:10]
# use the indices to get the corresponding paths
model_paths5 = [model_paths5[i] for i in indices3]
print(len(model_paths5),len(model_paths3),len(model_paths4))
model_paths = [model_paths5, model_paths3,model_paths4]



legend_labels = ['minacc', 'midacc','maxacc']

# 使用subplots创建3个子图和一个额外的空间给颜色轴
fig, axs = plt.subplots(1, 3, dpi=300, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 1, 0.1]})
cbar_ax = axs[-1]  # 最后一个axis用于颜色轴
cmap = plt.cm.jet
# 定义需要画图的权重层
layers = ["conv1.weight", "conv2.weight"]

final_epoch_accuracy_values_all = []

for ax_index, layer in enumerate(layers):
    ax = axs[ax_index]
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel(f'{layer} Mean Difference', fontsize=10)
    ax.tick_params(axis='both', labelsize=8)

    all_model_paths = [model_paths5,model_paths3, model_paths4]

    diff_values_all = []
    accuracies_all = []

    for idx, model_paths in enumerate(all_model_paths):
        diff_values_for_each_path = []
        final_epoch_accuracy_values = []

        for i, path in enumerate(model_paths):
            weight_generator = extract_weights(path, layer)
            experiment_weight_mean_values = []
            for k in range(1, 21):
                _, tensor1 = next(weight_generator)
                z_values = weightmean(tensor1)
                if isinstance(z_values, float):
                    z_values = [z_values]
                z_values = np.array(z_values)
                z_values = z_values[np.isfinite(z_values)]
                experiment_weight_mean_values.extend(z_values.tolist())

            # 计算每个路径的 epoch 之间的平均值差异
            diff_values = np.diff(experiment_weight_mean_values)
            diff_values_for_each_path.append(diff_values)

            # 从历史记录中获取最后一个 epoch 的准确率
            history_path = path.replace('weights.txt', 'train_history_dic.pkl')
            with open(history_path, 'rb') as g:
                history = torch.load(g)
            final_epoch_accuracy = history['model_epoch20']['test_acc']
            final_epoch_accuracy_values.append(final_epoch_accuracy)

        # 将每个路径的差异值组合
        diff_values_all.append(np.array(diff_values_for_each_path))
        # 计算该组的平均准确率
        accuracies_all.append(np.mean(final_epoch_accuracy_values))

    # 使用不同的颜色绘制每个类别的平均差异，并添加阴影
    for idx, diff_values in enumerate(diff_values_all):
        c = cmap(accuracies_all[idx])
        mean_diff_values = np.mean(diff_values, axis=0)
        min_diff_values = np.min(diff_values, axis=0)
        max_diff_values = np.max(diff_values, axis=0)
        ax.plot(range(2, 21), mean_diff_values, color=c, alpha=0.7, label=legend_labels[idx])
        ax.fill_between(range(2, 21), min_diff_values, max_diff_values, color=c, alpha=0.3)

    ax.set_xticks(range(1, 21))
    ax.set_xticklabels(range(1, 21))
    ax.legend(loc='best')

# 设置颜色轴
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Accuracy', rotation=270, labelpad=20)  # 这行添加了颜色轴的标签

# 适应布局
plt.tight_layout()

# 保存图形
plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result-cnn64-7-plot/weightdiff.png", dpi=300)

# 显示图形
plt.show()
