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

def count_weights_greater_than_zero(dictionary):
    tensor1 = dictionary.flatten()
    # Create a boolean mask where true values correspond to weights > 0
    mask = tensor1 > 0
    # Sum up the mask to get the number of weights > 0
    count = mask.sum().item()
    return count

def count_weights_lower_than_zero(dictionary):
    tensor1 = dictionary.flatten()
    # Create a boolean mask where true values correspond to weights > 0
    mask = tensor1 < 0
    # Sum up the mask to get the number of weights > 0
    count = mask.sum().item()
    return count

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
for i in range(1, 101):
    for j in range(1, 11):
        path = "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-mnist/b100l5lr0001-{}/train_result{}/weights.txt".format(i, j)
        model_paths.append(path)
        
model_paths1 = []
model_paths2 = []
model_paths3 = []
model_paths4 = []
test_acc_diff1 = []
test_acc_diff2 = []
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
    diff1 = abs(test_acc - 0.55)
    # append the path and the difference to the corresponding lists
    model_paths3.append(path)
    test_acc_diff1.append(diff1)

    diff2 = abs(test_acc - 0.9)
    # append the path and the difference to the corresponding lists
    model_paths4.append(path)
    test_acc_diff2.append(diff2)
# get the indices of the 100 smallest differences
indices1 = sorted(range(len(test_acc_diff1)), key=lambda i: test_acc_diff1[i])[:100]
# use the indices to get the corresponding paths
model_paths3 = [model_paths3[i] for i in indices1]
# get the indices of the 100 smallest differences
indices2 = sorted(range(len(test_acc_diff2)), key=lambda i: test_acc_diff2[i])[:100]
# use the indices to get the corresponding paths
model_paths4 = [model_paths4[i] for i in indices2]
print(len(model_paths1),len(model_paths2),len(model_paths3),len(model_paths4))
model_paths = [model_paths1, model_paths2, model_paths3, model_paths4]



legend_labels = ['minacc', 'midacc', 'maxacc']

# 使用subplots创建3个子图和一个额外的空间给颜色轴
fig, axs = plt.subplots(1, 4, dpi=300, figsize=(20, 6), gridspec_kw={'width_ratios': [1, 1, 1, 0.1]})
cbar_ax = axs[-1]  # 最后一个axis用于颜色轴
cmap = plt.cm.jet
# 定义需要画图的权重层
layers = ["fc1.weight", "fc2.weight", "fc3.weight"]

final_epoch_accuracy_values_all = []

for ax_index, layer in enumerate(layers):
    ax = axs[ax_index]

    # Other plot settings
    ax.set_xlabel('Epoch', fontsize=10)
    # ax.set_ylabel(f'{layer} postive number', fontsize=10)
    ax.set_ylabel(f'{layer} negative number', fontsize=10)
    ax.tick_params(axis='both', labelsize=8)

    # Group all model paths
    all_model_paths = [model_paths1,model_paths3,model_paths4]

    for idx, model_paths in enumerate(all_model_paths):
        all_weight_mean_values = []

        # Track all the epoch values (assuming they are the same for all paths)
        epoch_values = []
        final_epoch_accuracy_values = []

        for i, path in enumerate(model_paths):  
            weight_generator = extract_weights(path, layer)
            experiment_weight_mean_values = []
            for k in range(1, 21):
                epoch, tensor1 = next(weight_generator)
                if i == 0:  # Only on the first iteration, we save the epoch values
                    epoch_values.append(epoch)
                # z_values= count_weights_greater_than_zero(tensor1)
                z_values= count_weights_greater_than_zero(tensor1)
                if isinstance(z_values, float):
                    z_values = [z_values]
                # Convert to numpy array and filter out inf values
                z_values = np.array(z_values)
                z_values = z_values[np.isfinite(z_values)]
                experiment_weight_mean_values.extend(z_values.tolist())
            # Only add non-empty disparity values
            if experiment_weight_mean_values:
                all_weight_mean_values.append(experiment_weight_mean_values)

            # Get final epoch test accuracy
            history_path = path.replace('weights.txt', 'train_history_dic.pkl')
            with open(history_path, 'rb') as g:
                history = torch.load(g)
            final_epoch_accuracy = history['model_epoch20']['test_acc'] # Assuming 'model_epoch20' is always present
            final_epoch_accuracy_values.append(final_epoch_accuracy)
            final_epoch_accuracy_values_all.append(final_epoch_accuracy)

        # Convert to numpy array for easier manipulation
        all_weight_mean_values = np.array(all_weight_mean_values)

        # Calculate the minimum, maximum and mean node disparity values for each epoch
        min_values = np.min(all_weight_mean_values, axis=0)
        max_values = np.max(all_weight_mean_values, axis=0)
        mean_values = np.mean(all_weight_mean_values, axis=0)

        # Determine color from average final epoch accuracy
        c = cmap(np.mean(final_epoch_accuracy_values))

        # Create shaded range graph
        ax.fill_between(epoch_values, min_values, max_values, color=c, alpha=0.3)

        # Plot the mean line and add label for legend
        ax.plot(epoch_values, mean_values, color=c, alpha=0.7, label=legend_labels[idx])
        ax.set_xticks(range(1, 21))  # 设置 x 轴刻度
        ax.set_xticklabels(range(1, 21))  # 设置 x 轴刻度标签
    # Add the legend
    ax.legend(loc='best')

# 设置颜色轴
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Accuracy', rotation=270, labelpad=20)  # 这行添加了颜色轴的标签

# 适应布局
plt.tight_layout()

# 保存图形
plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-mnist-plot/positiveweightnum3clus_lineplot_all.png", dpi=300)

# 显示图形
plt.show()
