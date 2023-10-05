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
torch.set_printoptions(threshold=np.inf) #Display all data for complex matrix 

added_labels = []
num_subplots = 3
handles_list = [[] for _ in range(num_subplots)]
labels_list = [[] for _ in range(num_subplots)]

def scatter_with_accuracy(x, y, accuracy, ax, vmin, vmax, accuracy_type):
    global added_labels, handles_list, labels_list
    
    marker_dict = {
        'min accuracy': 'o',
        'mid accuracy': '^',
        'max accuracy': '*'
    }
    
    if accuracy_type not in added_labels:
        im = ax.scatter(x, y, c=accuracy, cmap='coolwarm', vmin=vmin, vmax=vmax, marker=marker_dict[accuracy_type])
        im = ax.scatter(x, y, color='salmon', marker=marker_dict[accuracy_type], s=10,label=accuracy_type)
        added_labels.append(accuracy_type)
    else:
        im = ax.scatter(x, y, c=accuracy, cmap='coolwarm', vmin=vmin, vmax=vmax, marker=marker_dict[accuracy_type])

    if len(added_labels) == 3:
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

def getgradient(dod,name):
    gradient=(dod[name])
    start_index = gradient.find("[[")
    end_index = gradient.rfind("]]") + 2
    desired_string = gradient[start_index:end_index]
    # define the string representation of the array
    # create a numpy array from the string representation
    array = np.array(eval(desired_string))
    # convert the numpy array to a tensor
    tensor = torch.from_numpy(array)  
    return tensor

def addweight(axs, path1, path2, vmin, vmax,flag):
    f = open(path1,'rb')
    dod1 = torch.load(f)
    torch.set_printoptions(threshold=np.inf) #Display all data for complex matrix     
    f = open(path2,'rb')
    dod2 = torch.load(f)
    torch.set_printoptions(threshold=np.inf) #Display all data for complex matrix
    tensor1=getgradient(dod1,"model_epoch20 fc1.weight").flatten()

    accuracy = dod2['model_epoch20']['test_acc']
    fc1_positive = copy.deepcopy(tensor1)
    fc1_negative = copy.deepcopy(tensor1)
    fc1_positive_mean = fc1_positive[fc1_positive > 0].mean()
    fc1_positive_std = fc1_positive[fc1_positive > 0].std()
    
    fc1_negative_mean = fc1_negative[fc1_negative < 0].mean()
    fc1_negative_std = fc1_negative[fc1_negative < 0].std()
    
    tensor2 = getgradient(dod1,"model_epoch20 fc2.weight").flatten()
    fc2_positive = copy.deepcopy(tensor2)
    fc2_negative = copy.deepcopy(tensor2)
    fc2_positive_mean = fc2_positive[fc2_positive > 0].mean()
    fc2_positive_std = fc2_positive[fc2_positive > 0].std()
    
    fc2_negative_mean = fc2_negative[fc2_negative < 0].mean()
    fc2_negative_std = fc2_negative[fc2_negative < 0].std()

    tensor3 = getgradient(dod1,"model_epoch20 fc3.weight").flatten()
    fc3_positive = copy.deepcopy(tensor3)
    fc3_negative = copy.deepcopy(tensor3)
    fc3_positive_mean = fc3_positive[fc3_positive > 0].mean()
    fc3_positive_std = fc3_positive[fc3_positive > 0].std()
    fc3_negative_mean = fc3_negative[fc3_negative < 0].mean()
    fc3_negative_std = fc3_negative[fc3_negative < 0].std()
    
    all_positive_mean = torch.cat((fc1_positive, fc2_positive, fc3_positive), dim=0).flatten().mean()
    all_negative_mean = torch.cat((fc1_negative, fc2_negative, fc3_negative), dim=0).flatten().mean()
    all_positive_std = torch.cat((fc1_positive, fc2_positive, fc3_positive), dim=0).flatten().std()
    all_negative_std = torch.cat((fc1_negative, fc2_negative, fc3_negative), dim=0).flatten().std()
    
    if flag==1:
        scatter_with_accuracy(fc1_positive_mean, fc1_positive_std, accuracy, axs[0], vmin, vmax,accuracy_type='min accuracy')
        scatter_with_accuracy(fc2_positive_mean, fc2_positive_std, accuracy, axs[1], vmin, vmax,accuracy_type='min accuracy')
        scatter_with_accuracy(fc3_positive_mean, fc3_positive_std, accuracy, axs[2], vmin, vmax,accuracy_type='min accuracy') 
        scatter_with_accuracy(all_positive_mean, all_positive_std, accuracy, axs[3], vmin, vmax,accuracy_type='min accuracy') 
    if flag==2:
        scatter_with_accuracy(fc1_positive_mean, fc1_positive_std, accuracy, axs[0], vmin, vmax,accuracy_type='mid accuracy')
        scatter_with_accuracy(fc2_positive_mean, fc2_positive_std, accuracy, axs[1], vmin, vmax,accuracy_type='mid accuracy')
        scatter_with_accuracy(fc3_positive_mean, fc3_positive_std, accuracy, axs[2], vmin, vmax,accuracy_type='mid accuracy')
        scatter_with_accuracy(all_positive_mean, all_positive_std, accuracy, axs[3], vmin, vmax,accuracy_type='mid accuracy') 
    else:
        scatter_with_accuracy(fc1_positive_mean, fc1_positive_std, accuracy, axs[0], vmin, vmax,accuracy_type='max accuracy')
        scatter_with_accuracy(fc2_positive_mean, fc2_positive_std, accuracy, axs[1], vmin, vmax,accuracy_type='max accuracy')
        scatter_with_accuracy(fc3_positive_mean, fc3_positive_std, accuracy, axs[2], vmin, vmax,accuracy_type='max accuracy')
        scatter_with_accuracy(all_positive_mean, all_positive_std, accuracy, axs[3], vmin, vmax,accuracy_type='max accuracy')

    # if flag==1:
    #     scatter_with_accuracy(fc1_negative_mean, fc1_negative_std, accuracy, axs[0], vmin, vmax,accuracy_type='min accuracy')
    #     scatter_with_accuracy(fc2_negative_mean, fc2_negative_std, accuracy, axs[1], vmin, vmax,accuracy_type='min accuracy')
    #     scatter_with_accuracy(fc3_negative_mean, fc3_negative_std, accuracy, axs[2], vmin, vmax,accuracy_type='min accuracy') 
    #     scatter_with_accuracy(all_negative_mean, all_negative_std, accuracy, axs[3], vmin, vmax,accuracy_type='min accuracy') 
    # if flag==2:
    #     scatter_with_accuracy(fc1_negative_mean, fc1_negative_std, accuracy, axs[0], vmin, vmax,accuracy_type='mid accuracy')
    #     scatter_with_accuracy(fc2_negative_mean, fc2_negative_std, accuracy, axs[1], vmin, vmax,accuracy_type='mid accuracy')
    #     scatter_with_accuracy(fc3_negative_mean, fc3_negative_std, accuracy, axs[2], vmin, vmax,accuracy_type='mid accuracy')
    #     scatter_with_accuracy(all_negative_mean, all_negative_std, accuracy, axs[3], vmin, vmax,accuracy_type='mid accuracy') 
    # else:
    #     scatter_with_accuracy(fc1_negative_mean, fc1_negative_std, accuracy, axs[0], vmin, vmax,accuracy_type='max accuracy')
    #     scatter_with_accuracy(fc2_negative_mean, fc2_negative_std, accuracy, axs[1], vmin, vmax,accuracy_type='max accuracy')
    #     scatter_with_accuracy(fc3_negative_mean, fc3_negative_std, accuracy, axs[2], vmin, vmax,accuracy_type='max accuracy')
    #     scatter_with_accuracy(all_negative_mean, all_negative_std, accuracy, axs[3], vmin, vmax,accuracy_type='max accuracy')
    

fig, axs = plt.subplots(nrows=1, ncols=4, dpi=300,figsize=(15, 3))

# define the minimum and maximum values for the colorbar
vmin = 0.0
vmax = 1.0

# loop through each set of files to plot
# for i in range(100):
#     for j in range(10):
#         path1 = f"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-fashionmnist/b100l5lr0001-{i+1}/train_result{j+1}/gradient_dic.pkl"
#         path2 = f"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-fashionmnist/b100l5lr0001-{i+1}/train_result{j+1}/train_history_dic.pkl"
#         addweight(axs, path1, path2, vmin, vmax,1)


model_paths = []    
for i in range(1, 101):
    for j in range(1, 11):
        path = "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-fashionmnist/b100l5lr0001-{}/train_result{}/gradient_dic.pkl".format(i, j)
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
        if test_acc < 0.2:
            model_paths1.append(path)
        # Check if the test accuracy is in the range [0.2, 0.5)
        if 0.2 <= test_acc < 0.55:
            model_paths2.append(path)
        # compute the difference between test_acc and 0.65
        diff1 = abs(test_acc - 1)
        # append the path and the difference to the corresponding lists
        model_paths3.append(path)
        test_acc_diff1.append(diff1)
# get the indices of the 100 smallest differences
indices1 = sorted(range(len(test_acc_diff1)), key=lambda i: test_acc_diff1[i])[:100]
# use the indices to get the corresponding paths
model_paths3 = [model_paths3[i] for i in indices1]
print(len(model_paths1),len(model_paths2),len(model_paths3))

for path in model_paths1:
    path1 = path
    path2 = path.replace('gradient_dic.pkl', 'train_history_dic.pkl')
    addweight(axs, path1, path2, vmin, vmax, 1)
    
for path in model_paths2:
    path1 = path
    path2 = path.replace('gradient_dic.pkl', 'train_history_dic.pkl')
    addweight(axs, path1, path2, vmin, vmax, 2)

for path in model_paths3:
    path1 = path
    path2 = path.replace('gradient_dic.pkl', 'train_history_dic.pkl')
    addweight(axs, path1, path2, vmin, vmax, 3)

# set the colorbar
cbar = fig.colorbar(axs[0].collections[0], ax=axs, orientation='vertical', shrink=0.75)
cbar.ax.set_ylabel('Accuracy')
plt.tight_layout()
plt.subplots_adjust(right=0.75)

plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-fashionmnist-plot/mean_std_scatter_positive_gradient_minmax.png", dpi=300)
plt.show()
