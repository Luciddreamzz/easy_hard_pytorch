import numpy as np 
import matplotlib.mlab as mlab 
 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import torch
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

import seaborn as sns 
import matplotlib as mpl 
from torch import tensor
torch.set_printoptions(threshold=np.inf) #Display all data for complex matrix 显示所有省略号的玩意

added_labels = []
num_subplots = 3
handles_list = [[] for _ in range(num_subplots)]
labels_list = [[] for _ in range(num_subplots)]

def scatter_with_accuracy(x, y, accuracy, ax, vmin, vmax, accuracy_type):
    global added_labels, handles_list, labels_list
    
    marker_dict = {
        'min accuracy': 'o',
        # 'mid accuracy': '^',
        'max accuracy': '*'
    }
    
    if accuracy_type not in added_labels:
        im = ax.scatter(x, y, c=accuracy, cmap='jet', vmin=vmin, vmax=vmax, marker=marker_dict[accuracy_type])
        im = ax.scatter(x, y, color='blue', marker=marker_dict[accuracy_type], s=10,label=accuracy_type)
        added_labels.append(accuracy_type)
    else:
        im = ax.scatter(x, y, c=accuracy, cmap='jet', vmin=vmin, vmax=vmax, marker=marker_dict[accuracy_type])

    if len(added_labels) == 2:
        for i in range(num_subplots):
            handles, labels = handles_list[i], labels_list[i]
            if not handles:
                handles_list[i], labels_list[i] = ax.get_legend_handles_labels()
                ax.legend(handles=handles_list[i], labels=labels_list[i], loc='best', fontsize=4)
            else:
                ax.legend(handles=handles, labels=labels, loc='best', fontsize=4)


    ax.tick_params(axis='both', labelsize=6)
    ax.set_xlabel('mean',fontsize=7)
    ax.set_ylabel('std',fontsize=7)

    # axs[0].set_ylim(-2, 6) 
    # axs[1].set_ylim(-8, 1) 
    # axs[2].set_ylim(-8, 1) 
    # axs[0].set_xlim(-80,20) 
    # axs[1].set_xlim(-2, 6) 
    # axs[2].set_xlim(-80,20)
    ax.tick_params(axis='both', labelsize=6)
    return im

def csvfilesave(csvfile_path,list_name):
    csvFile = open(csvfile_path, "a+",)
    try:
        writer = csv.writer(csvFile)
        writer.writerow(list_name)
    finally:
        csvFile.close() 

def addgradient(axs, path1, path2, vmin, vmax,flag):
    f1 = open(path1,'rb')
    dod1 = torch.load(f1)
    f2 = open(path2,'rb')
    dod2 = torch.load(f2)
    torch.set_printoptions(threshold=np.inf) #Display all data for complex matrix
    tensor1=(dod1["model_epoch20 fc1.weight"])
    tensor2=(dod1["model_epoch20 fc2.weight"])
    tensor3=(dod1["model_epoch20 fc3.weight"])
    accuracy = dod2['model_epoch20']['test_acc']
    start_index1 = tensor1.find("[[")
    end_index1 = tensor1.rfind("]]") + 2
    desired_string1 = tensor1[start_index1:end_index1]
    # define the string representation of the array
    # create a numpy array from the string representation
    array1 = np.array(eval(desired_string1))
    # convert the numpy array to a tensor
    gradientfc1 = (torch.from_numpy(array1)).flatten()
    
    start_index2 = tensor2.find("[[")
    end_index2 = tensor2.rfind("]]") + 2
    desired_string2 = tensor2[start_index2:end_index2]
    # define the string representation of the array
    # create a numpy array from the string representation
    array2 = np.array(eval(desired_string2))
    # convert the numpy array to a tensor
    gradientfc2 = (torch.from_numpy(array2)).flatten()
    
    start_index3 = tensor3.find("[[")
    end_index3 = tensor3 .rfind("]]") + 2
    desired_string3 = tensor3[start_index3:end_index3]
    # define the string representation of the array
    # create a numpy array from the string representation
    array3  = np.array(eval(desired_string3))
    # convert the numpy array to a tensor
    gradientfc3 = (torch.from_numpy(array3)).flatten()
    
    fc1_mean=gradientfc1.mean()
    fc2_mean=gradientfc2.mean()
    fc3_mean=gradientfc3.mean()
    all_mean = torch.cat((gradientfc1, gradientfc2, gradientfc3), dim=0).mean()
    fc1_std=gradientfc1.std()
    fc2_std=gradientfc2.std()
    fc3_std=gradientfc3.std()
    all_std = torch.cat((gradientfc1, gradientfc2, gradientfc3), dim=0).std()

    
    if flag==1:
        scatter_with_accuracy(fc1_mean, fc1_std, accuracy, axs[0], vmin, vmax,accuracy_type='min accuracy')
        scatter_with_accuracy(fc2_mean, fc2_std, accuracy, axs[1], vmin, vmax,accuracy_type='min accuracy')
        scatter_with_accuracy(fc3_mean, fc3_std, accuracy, axs[2], vmin, vmax,accuracy_type='min accuracy') 
        scatter_with_accuracy(all_mean, all_std, accuracy, axs[3], vmin, vmax,accuracy_type='min accuracy') 
    # if flag==2:
    #     scatter_with_accuracy(fc1_mean, fc1_std, accuracy, axs[0], vmin, vmax,accuracy_type='mid accuracy')
    #     scatter_with_accuracy(fc2_mean, fc2_std, accuracy, axs[1], vmin, vmax,accuracy_type='mid accuracy')
    #     scatter_with_accuracy(fc3_mean, fc3_std, accuracy, axs[2], vmin, vmax,accuracy_type='mid accuracy')
    #     scatter_with_accuracy(all_mean, all_std, accuracy, axs[3], vmin, vmax,accuracy_type='mid accuracy') 
    else:
        scatter_with_accuracy(fc1_mean, fc1_std, accuracy, axs[0], vmin, vmax,accuracy_type='max accuracy')
        scatter_with_accuracy(fc2_mean, fc2_std, accuracy, axs[1], vmin, vmax,accuracy_type='max accuracy')
        scatter_with_accuracy(fc3_mean, fc3_std, accuracy, axs[2], vmin, vmax,accuracy_type='max accuracy')
        scatter_with_accuracy(all_mean, all_std, accuracy, axs[3], vmin, vmax,accuracy_type='max accuracy')

fig, axs = plt.subplots(nrows=1, ncols=4, dpi=300,figsize=(15, 3))
# define the minimum and maximum values for the colorbar
vmin = 0.2
vmax = 0.4

# loop through each set of files to plot
# for i in range(100):
#     for j in range(10):
#         path1 = f"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-cifar10/b100l5lr0001-{i+1}/train_result{j+1}/gradient_dic.pkl"
#         path2 = f"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-cifar10/b100l5lr0001-{i+1}/train_result{j+1}/train_history_dic.pkl"
#         addgradient(axs, path1, path2, vmin, vmax,1)


model_paths = []    
for i in range(1, 101):
    for j in range(1, 11):
        path = "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-cifar10/b100l5lr0001-{}/train_result{}/gradient_dic.pkl".format(i, j)
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
    with open(path, 'rb') as f:
        # Load cnnmodel and history dictionary from the file
        cnnmodel = torch.load(f)
        history_path = path.replace('gradient_dic.pkl', 'train_history_dic.pkl')
        with open(history_path, 'rb') as g:
            history = torch.load(g)
        # Get test accuracy from the history dictionary
        test_acc = history['model_epoch20']['test_acc']
        diff1 = abs(test_acc - 0)
        # append the path and the difference to the corresponding lists
        model_paths3.append(path)
        test_acc_diff1.append(diff1)
        diff2 = abs(test_acc - 1)
        # append the path and the difference to the corresponding lists
        model_paths4.append(path)
        test_acc_diff2.append(diff2)
# get the indices of the 100 smallest differences
indices1 = sorted(range(len(test_acc_diff1)), key=lambda i: test_acc_diff1[i])[:100]
# use the indices to get the corresponding paths
model_paths3 = [model_paths3[i] for i in indices1]
indices2 = sorted(range(len(test_acc_diff2)), key=lambda i: test_acc_diff2[i])[:100]
# use the indices to get the corresponding paths
model_paths4 = [model_paths4[i] for i in indices2]
print(len(model_paths3),len(model_paths4))

for path in model_paths3:
    path1 = path
    path2 = path.replace('gradient_dic.pkl', 'train_history_dic.pkl')
    addgradient(axs, path1, path2, vmin, vmax, 1)
    
# for path in model_paths2:
#     path1 = path
#     path2 = path.replace('gradient_dic.pkl', 'train_history_dic.pkl')
#     addgradient(axs, path1, path2, vmin, vmax, 2)

for path in model_paths4:
    path1 = path
    path2 = path.replace('gradient_dic.pkl', 'train_history_dic.pkl')
    addgradient(axs, path1, path2, vmin, vmax, 2)


# set the colorbar
cbar = fig.colorbar(axs[0].collections[0], ax=axs, orientation='vertical', shrink=0.75)
cbar.ax.set_ylabel('Accuracy')
plt.tight_layout()
plt.subplots_adjust(right=0.75)
plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-cifar10plot/mean_std_scatter_gradient_minmax.png", dpi=300)
plt.show()