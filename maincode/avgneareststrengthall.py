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
        'lowermid accuracy': '^',
        'highermid accuracy': 's',
        'max accuracy': '*'
    }
    
    if accuracy_type not in added_labels:
        im = ax.scatter(x, y, c=accuracy, cmap='RdPu', vmin=vmin, vmax=vmax, marker=marker_dict[accuracy_type], label=accuracy_type)
        added_labels.append(accuracy_type)
    else:
        im = ax.scatter(x, y, c=accuracy, cmap='RdPu', vmin=vmin, vmax=vmax, marker=marker_dict[accuracy_type])

    if len(added_labels) == 4:
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


    axs[0].set_xlim(-500, 10000)
    axs[0].set_ylim(-500, 20000)
    axs[1].set_ylim(0, 2)
    axs[1].set_xlim(-100,1000)
    axs[2].set_xlim(-100, 1000)
    axs[2].set_ylim(0, 2)
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
        scatter_with_accuracy(z1, z2, accuracy, axs[0], vmin, vmax,accuracy_type='lowermid accuracy')
        scatter_with_accuracy(z2, z3, accuracy, axs[1], vmin, vmax,accuracy_type='lowermid accuracy')
        scatter_with_accuracy(z1, z3, accuracy, axs[2], vmin, vmax,accuracy_type='lowermid accuracy')
    if flag==3:
        scatter_with_accuracy(z1, z2, accuracy, axs[0], vmin, vmax,accuracy_type='highermid accuracy')
        scatter_with_accuracy(z2, z3, accuracy, axs[1], vmin, vmax,accuracy_type='highermid accuracy')
        scatter_with_accuracy(z1, z3, accuracy, axs[2], vmin, vmax,accuracy_type='highermid accuracy')
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
# loop through each set of files to plot
for i in range(100):
    for j in range(10):
        path1 = f"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-mnist/b100l5lr0001-{i+1}/train_result{j+1}/cnnmodel.pkl"
        path2 = f"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-mnist/b100l5lr0001-{i+1}/train_result{j+1}/train_history_dic.pkl"
        addweight(axs, path1, path2, vmin, vmax,1)

# Define the paths of the trained models
# min
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result8/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result8/train_history_dic.pkl", vmin, vmax,1)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result9/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result9/train_history_dic.pkl", vmin, vmax,1)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result10/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result10/train_history_dic.pkl", vmin, vmax,1)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result1/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result1/train_history_dic.pkl", vmin, vmax,1)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result6/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result6/train_history_dic.pkl", vmin, vmax,1)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result6/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result6/train_history_dic.pkl", vmin, vmax,1)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result3/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result3/train_history_dic.pkl", vmin, vmax,1)
# lowermid
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result7/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result7/train_history_dic.pkl", vmin, vmax,2)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result3/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result3/train_history_dic.pkl", vmin, vmax,2)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result1/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result1/train_history_dic.pkl", vmin, vmax,2)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result3/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result3/train_history_dic.pkl", vmin, vmax,2)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result2/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result2/train_history_dic.pkl", vmin, vmax,2)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result4/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result4/train_history_dic.pkl", vmin, vmax,2)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-8/train_result2/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-8/train_result2/train_history_dic.pkl", vmin, vmax,2)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result8/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result8/train_history_dic.pkl", vmin, vmax,2)
# # highermid
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result10/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result10/train_history_dic.pkl", vmin, vmax,3)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result9/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result9/train_history_dic.pkl", vmin, vmax,3)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result4/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result4/train_history_dic.pkl", vmin, vmax,3)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result5/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result5/train_history_dic.pkl", vmin, vmax,3)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result8/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result8/train_history_dic.pkl", vmin, vmax,3)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result10/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result10/train_history_dic.pkl", vmin, vmax,3)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result8/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result8/train_history_dic.pkl", vmin, vmax,3)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result9/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result9/train_history_dic.pkl", vmin, vmax,3)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result1/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result1/train_history_dic.pkl", vmin, vmax,3)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result10/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result10/train_history_dic.pkl", vmin, vmax,3)
# # max
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result5/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result5/train_history_dic.pkl", vmin, vmax,0)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result6/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result6/train_history_dic.pkl", vmin, vmax,0)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result5/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result5/train_history_dic.pkl", vmin, vmax,0)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result5/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result5/train_history_dic.pkl", vmin, vmax,0)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result6/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result6/train_history_dic.pkl", vmin, vmax,0)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result8/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result8/train_history_dic.pkl", vmin, vmax,0)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result10/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result10/train_history_dic.pkl", vmin, vmax,0)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result2/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result2/train_history_dic.pkl", vmin, vmax,0)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result2/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result2/train_history_dic.pkl", vmin, vmax,0)
# addweight(axs,"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result10/cnnmodel.pkl", "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result10/train_history_dic.pkl", vmin, vmax,0)

# set the colorbar
cbar = fig.colorbar(axs[0].collections[0], ax=axs, orientation='vertical', shrink=0.75)
cbar.ax.set_ylabel('Accuracy')
# Add the legend

plt.tight_layout()
plt.subplots_adjust(right=0.75)
plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-mnist-plot/avgneareststrengthall.png", dpi=300)
plt.show()


