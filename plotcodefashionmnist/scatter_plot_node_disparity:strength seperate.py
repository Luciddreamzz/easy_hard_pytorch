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

# nodestrengthall
    # axs[0].set_ylim(-2, 6) 
    # axs[1].set_ylim(-8, 1) 
    # axs[2].set_ylim(-8, 1) 
    # axs[0].set_xlim(-80,20) 
    # axs[1].set_xlim(-2, 6) 
    # axs[2].set_xlim(-80,20)
# nodestrengthminmax
    # axs[0].set_ylim(-1, 6) 
    # axs[1].set_ylim(-6, 1) 
    # axs[2].set_ylim(-6, 1) 
    # axs[0].set_xlim(-70,20) 
    # axs[1].set_xlim(-1, 6) 
    # axs[2].set_xlim(-70,20)
# nodedisparityminmax    
    axs[0].set_ylim(-20, 500)
    axs[0].set_xlim(-20, 500)
    axs[1].set_ylim(-500, 10000)
    axs[1].set_xlim(-20, 500)
    axs[2].set_ylim(-500, 10000)
    axs[2].set_xlim(-50, 1000)
    ax.tick_params(axis='both', labelsize=6)
    return im



#---------------------------------Property of the neural nodes----------------------------------------------------
#If the disparity of output edges, the order of layers should be reversed e.g.(set layer1 in layer2 position, set layer2 in layer 1 position)
def nodesproperty(dictionary,layer1,layer2,flag):
    z_sum=0
    z1=[]
    if flag==1:
        s=dictionary.sum(axis=1)#For input egdes, find the sum of the rows
        for i in range (0,layer2):
            for j in range (0,layer1):
                z=(abs(dictionary[i][j])/abs(s[i]))**2 
                z_sum+=z  
            z1.append('{:.4f}'.format(z_sum))
            z_sum=0
    else:
        s=dictionary.sum(axis=0)#For output edges, sum the columns
        for i in range (0,layer2):
            for j in range (0,layer1):
                z=(abs(dictionary[j][i])/abs(s[i]))**2 
                z_sum+=z   
            z1.append('{:.4f}'.format(z_sum))
            z_sum=0
    return sum([float(x) if x != 'nan' else 0 for x in z1]),torch.sum(s).item()

def addweight(axs, path1, path2, vmin, vmax,flag):
    f = open(path1,'rb')
    dod1 = torch.load(f)
    f = open(path2,'rb')
    dod2 = torch.load(f)
    torch.set_printoptions(threshold=np.inf) #Display all data for complex matrix
    
    tensor1 = (dod1["fc1.weight"])
    tensor2 = (dod1["fc2.weight"])
    tensor3 = (dod1["fc3.weight"])
    accuracy = dod2['model_epoch20']['test_acc']
    

    z1,s1=nodesproperty(tensor1,input_size,node1,1)
    z2,s2=nodesproperty(tensor2,node1,node2,1)
    z3,s3=nodesproperty(tensor3,node2,output_size,1)
    
    fc1_positive = copy.deepcopy(tensor1)
    fc1_negative = copy.deepcopy(tensor1)
    fc1_positive[fc1_positive<0]=0
    fc1_negative[fc1_negative>0]=0
    
    fc2_positive = copy.deepcopy(tensor2)
    fc2_negative = copy.deepcopy(tensor2)
    fc2_positive[fc2_positive<0]=0
    fc2_negative[fc2_negative>0]=0

    fc3_positive = copy.deepcopy(tensor3)
    fc3_negative = copy.deepcopy(tensor3)
    fc3_positive[fc3_positive<0]=0
    fc3_negative[fc3_negative>0]=0
    
    #Disparity and strength of the neural nodes-- input edges --positive
    z4,s4=nodesproperty(fc1_positive,input_size,node1,1)
    z5,s5=nodesproperty(fc2_positive,node1,node2,1)
    z6,s6=nodesproperty(fc3_positive,node2,output_size,1)  

    #Disparity and strength of the neural nodes-- input edges --negative
    z7,s7=nodesproperty(fc1_negative,input_size,node1,1)
    z8,s8=nodesproperty(fc2_negative,node1,node2,1)
    z9,s9=nodesproperty(fc3_negative,node2,output_size,1)  
    
    #o1:Disparity of the neural nodes-- output edges
    #o2:hidden layer1-hidden layer2
    #o3:hidden layer2-output layer   
    #t1:Strength of the neural nodes-- output edges
    #t2:hidden layer1-hidden layer2
    #t3:hidden layer2-output layer
    o1,t1=nodesproperty(tensor1,node1,input_size,0)
    o2,t2=nodesproperty(tensor2,node2,node1,0)
    o3,t3=nodesproperty(tensor3,output_size,node2,0)
    #Disparity and strength of the neural nodes-- output edges --positive   
    o4,t4=nodesproperty(fc1_positive,node1,input_size,0)
    o5,t5=nodesproperty(fc2_positive,node2,node1,0)
    o6,t6=nodesproperty(fc3_positive,output_size,node2,0)
    #Disparity and strength of the neural nodes-- output edges --negative
    o7,t7=nodesproperty(fc1_negative,node1,input_size,0)
    o8,t8=nodesproperty(fc2_negative,node2,node1,0)
    o9,t9=nodesproperty(fc3_negative,output_size,node2,0)
    # plot sactter of input edges sum node disparity and sum node strength
    # if flag==1:
    #     scatter_with_accuracy(s1, s2, accuracy, axs[0], vmin, vmax,accuracy_type='min accuracy')
    #     scatter_with_accuracy(s2, s3, accuracy, axs[1], vmin, vmax,accuracy_type='min accuracy')
    #     scatter_with_accuracy(s1, s3, accuracy, axs[2], vmin, vmax,accuracy_type='min accuracy') 
    # if flag==2:
    #     scatter_with_accuracy(s1, s2, accuracy, axs[0], vmin, vmax,accuracy_type='mid accuracy')
    #     scatter_with_accuracy(s2, s3, accuracy, axs[1], vmin, vmax,accuracy_type='mid accuracy')
    #     scatter_with_accuracy(s1, s3, accuracy, axs[2], vmin, vmax,accuracy_type='mid accuracy')
    # else:
    #     scatter_with_accuracy(s1, s2, accuracy, axs[0], vmin, vmax,accuracy_type='max accuracy')
    #     scatter_with_accuracy(s2, s3, accuracy, axs[1], vmin, vmax,accuracy_type='max accuracy')
    #     scatter_with_accuracy(s1, s3, accuracy, axs[2], vmin, vmax,accuracy_type='max accuracy')
    
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
plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-fashionmnist-plot/nodedisparityminmax.png", dpi=300)
plt.show()


