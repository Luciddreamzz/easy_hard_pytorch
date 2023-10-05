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

def average_neighbor_strength(model_path, name):
    # 加载模型
    model_dict = torch.load(model_path)
    weight_matrix = model_dict[name]

    # 计算每个节点的出度和邻居节点的权重之和
    node_outdegree = weight_matrix.sum(dim=1)
    neighbor_strengths = torch.div(weight_matrix, node_outdegree.unsqueeze(1))

    # 计算每个节点的SWNN值
    swnn_values = torch.zeros_like(node_outdegree)
    for i in range(weight_matrix.shape[0]):
        neighbor_weights = neighbor_strengths[i]
        swnn_values[i] = torch.sum(weight_matrix[i] * neighbor_weights)

    return (swnn_values / node_outdegree).mean()

def scatter_with_accuracy(x, y, accuracy, ax, vmin, vmax, accuracy_type):
    global added_labels, handles_list, labels_list
    
    marker_dict = {
        'min accuracy': 'o',
        'mid accuracy': '^',
        'max accuracy': '*'
    }
    
    if accuracy_type not in added_labels:
        im = ax.scatter(x, y, c=accuracy, cmap='jet', vmin=vmin, vmax=vmax, marker=marker_dict[accuracy_type])
        im = ax.scatter(x, y, color='red', marker=marker_dict[accuracy_type], s=10,label=accuracy_type)
        added_labels.append(accuracy_type)
    else:
        im = ax.scatter(x, y, c=accuracy, cmap='jet', vmin=vmin, vmax=vmax, marker=marker_dict[accuracy_type])

    if len(added_labels) == 3:
        for i in range(num_subplots):
            handles, labels = handles_list[i], labels_list[i]
            if not handles:
                handles_list[i], labels_list[i] = ax.get_legend_handles_labels()
                ax.legend(handles=handles_list[i], labels=labels_list[i], loc='best', fontsize=4)
            else:
                ax.legend(handles=handles, labels=labels, loc='best', fontsize=4)


    ax.tick_params(axis='both', labelsize=6)

    axs[0].set_xlabel('input layer', fontsize=7)
    axs[0].set_ylabel('1st hidden layer', fontsize=7)
    axs[1].set_xlabel('1st hidden layer', fontsize=7)
    axs[1].set_ylabel('2nd hidden layer', fontsize=7)
    axs[2].set_xlabel('input layer', fontsize=7)
    axs[2].set_ylabel('2nd hidden layer', fontsize=7)

    #all 
    # axs[0].set_ylim(-20, 500)
    # axs[0].set_xlim(-20, 500)
    # axs[1].set_ylim(-500, 10000)
    # axs[1].set_xlim(-20, 500)
    # axs[2].set_ylim(-500, 10000)
    # axs[2].set_xlim(-50, 1000)
    
    #maxmin
    axs[0].set_ylim(-5, 150)
    axs[0].set_xlim(-5, 175)
    axs[1].set_ylim(-100, 3000)
    axs[1].set_xlim(-5, 150)
    axs[2].set_ylim(-100, 3000)
    axs[2].set_xlim(-5, 175)
    
    #maxminmid
    # axs[0].set_xlim(-0.1, 3)
    # axs[0].set_ylim(-2, 90)
    # axs[1].set_xlim(-5, 90)
    # axs[1].set_ylim(-50, 1500)
    # axs[2].set_xlim(-0.1, 3)
    # axs[2].set_ylim(-50, 2000)
    ax.tick_params(axis='both', labelsize=6)
    return im



#---------------------------------Property of the neural nodes----------------------------------------------------
#If the disparity of output edges, the order of layers should be reversed e.g.(set layer1 in layer2 position, set layer2 in layer 1 position)

def addweight(axs, path1, path2, vmin, vmax,flag):

    f = open(path2,'rb')
    dod2 = torch.load(f)
    torch.set_printoptions(threshold=np.inf) #Display all data for complex matrix

    accuracy = dod2['model_epoch20']['test_acc']
    

    z1 = average_neighbor_strength(path1,"fc1.weight")
    z2 = average_neighbor_strength(path1,"fc2.weight")
    z3 = average_neighbor_strength(path1,"fc3.weight")

    # plot sactter of input edges sum node disparity and sum node strength
    if flag==1:
        scatter_with_accuracy(z1, z2, accuracy, axs[0], vmin, vmax,accuracy_type='min accuracy')
        scatter_with_accuracy(z2, z3, accuracy, axs[1], vmin, vmax,accuracy_type='min accuracy')
        scatter_with_accuracy(z1, z3, accuracy, axs[2], vmin, vmax,accuracy_type='min accuracy')
    if flag==2:
        scatter_with_accuracy(z1, z2, accuracy, axs[0], vmin, vmax,accuracy_type='mid accuracy')
        scatter_with_accuracy(z2, z3, accuracy, axs[1], vmin, vmax,accuracy_type='mid accuracy')
        scatter_with_accuracy(z1, z3, accuracy, axs[2], vmin, vmax,accuracy_type='mid accuracy')
    else:
        scatter_with_accuracy(z1, z2, accuracy, axs[0], vmin, vmax,accuracy_type='max accuracy')
        scatter_with_accuracy(z2, z3, accuracy, axs[1], vmin, vmax,accuracy_type='max accuracy')
        scatter_with_accuracy(z1, z3, accuracy, axs[2], vmin, vmax,accuracy_type='max accuracy')

    

fig, axs = plt.subplots(nrows=1, ncols=3, dpi=300,figsize=(12, 3))

# define the minimum and maximum values for the colorbar
vmin = 0.0
vmax = 1.0
#hyperparameters of networks
input_size= 28*28
output_size=10
node1=5
node2=5
a=0
# loop through each set of files to plot
# for i in range(100):
#     for j in range(10):
#         path1 = f"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-fashionmnist/b100l5lr0001-{i+1}/train_result{j+1}/cnnmodel.pkl"
#         path2 = f"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-fashionmnist/b100l5lr0001-{i+1}/train_result{j+1}/train_history_dic.pkl"
#         addweight(axs, path1, path2, vmin, vmax,1)


model_paths = []    
for i in range(1, 101):
    for j in range(1, 11):
        path = "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-fashionmnist/b100l5lr0001-{}/train_result{}/cnnmodel.pkl".format(i, j)
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
        history_path = path.replace('cnnmodel.pkl', 'train_history_dic.pkl')
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
    path2 = path.replace('cnnmodel.pkl', 'train_history_dic.pkl')
    addweight(axs, path1, path2, vmin, vmax, 1)
    
for path in model_paths2:
    path1 = path
    path2 = path.replace('cnnmodel.pkl', 'train_history_dic.pkl')
    addweight(axs, path1, path2, vmin, vmax, 2)

for path in model_paths3:
    path1 = path
    path2 = path.replace('cnnmodel.pkl', 'train_history_dic.pkl')
    addweight(axs, path1, path2, vmin, vmax, 3)

# set the colorbar
cbar = fig.colorbar(axs[0].collections[0], ax=axs, orientation='vertical', shrink=0.75)
cbar.ax.set_ylabel('Accuracy')
# Add the legend

plt.tight_layout()
plt.subplots_adjust(right=0.75)
plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-fashionmnist-plot/avgneareststrengthminmax.png", dpi=300)
plt.show()


