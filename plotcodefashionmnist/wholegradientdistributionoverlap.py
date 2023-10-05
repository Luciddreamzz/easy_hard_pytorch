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
#---------------------------------Freedman-Diaconis规则用于确定直方图中的条柱宽度 The Freedman-Diaconis rule is used to determine the width of the bars in the histogram----------------------------------------------------
def freedman_diaconis_rule(data):
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    bin_width = 2 * iqr / len(data) ** (1/3)
    return bin_width
#---------------------------------把字符串转换成张量 let string transform to tensor----------------------------------------------------
def stringtotensor(path,name):
    #fig, ax = plt.subplots()
    f=open(path,'rb')
    dod=torch.load(f)
    torch.set_printoptions(threshold=np.inf) #Display all data for complex matrix 显示所有省略号的玩意
    gradient=dod[name]
    start_index = gradient.find("[[")
    end_index = gradient.rfind("]]") + 2
    desired_string = gradient[start_index:end_index]
    # define the string representation of the array
    # create a numpy array from the string representation
    array = np.array(eval(desired_string))
# convert the numpy array to a tensor
    tensor = torch.from_numpy(array)
    return tensor

#---------------------------------The Scott_rule is used to determine the width of the bars in the histogram----------------------------------------------------
def Scott_rule(data):
    bin_width = 3.5 * np.std(data.numpy()) / len(data) ** (1/3)
    print(bin_width)
    return bin_width

def getbins(fc_weight):
    bins=np.arange(min(fc_weight).item(), max(fc_weight).item() + Scott_rule(fc_weight), Scott_rule(fc_weight))
    return bins

def getbins1(fc_weight):
    bins=np.arange(min(fc_weight).item(), max(fc_weight).item() + 0.01, 0.01)
    return bins
#---------------------------------add all tensor into one tensor----------------------------------------------------
def tensoradd2(tensor1,tensor2,tensor3):
# concatenate the two tensors along dimension 0
    concatenated_tensor = torch.cat((tensor1,tensor2,tensor3), dim=0).flatten()
    return concatenated_tensor
#---------------------------------画图----------------------------------------------------
def painthistoverlap(fc_weight_max,fc_weight_mid,fc_weight_min,flag):
    
    plt.subplot(1,4,flag)
    #fig, ax = plt.subplots()
    sns.distplot(fc_weight_max,bins=getbins1(fc_weight_max),kde_kws={"color":"blueviolet", "lw":1 ,"linestyle":"-","label":"max accuracy"}, hist_kws={ "color": "blueviolet" }) #lightcoral mediumturquoise
    sns.distplot(fc_weight_mid,bins=getbins1(fc_weight_mid),kde_kws={"color":"deepskyblue", "lw":1 ,"linestyle":"-","label":"mid accuracy"}, hist_kws={ "color": "deepskyblue" })
    sns.distplot(fc_weight_min,bins=getbins1(fc_weight_min),kde_kws={"color":"lightcoral", "lw":1 ,"linestyle":"-","label":"min accuracy"}, hist_kws={ "color": "lightcoral" })
    plt.xticks(fontproperties = 'Times New Roman', size = 6)
    plt.yticks(fontproperties = 'Times New Roman', size = 6)
    plt.title("",fontsize=8)
    plt.xlabel("gradients",fontsize=6)
    plt.ylabel("Frequency",fontsize=6)
    plt.legend(loc='best',fontsize=6)
    # set the x and y limits for the current subplot
    if flag == 2:
        plt.xlim(xmin=-1, xmax=1)
        plt.ylim(ymin=0, ymax=12)
    elif flag == 3:
        plt.xlim(xmin=-2, xmax=2)
        plt.ylim(ymin=0, ymax=5)
    elif flag == 4:
        plt.xlim(xmin=-0.1, xmax=0.1)
        plt.ylim(ymin=0, ymax=80)


#---------------------------------main function---------------------------------------------------- 

plt.figure(figsize=(15,3),dpi=300)
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
#---------------------------------get gradient----------------------------------------------------
max_fc1_weight_sum,max_fc2_weight_sum,max_fc3_weight_sum = [],[],[]
mid_fc1_weight_sum,mid_fc2_weight_sum,mid_fc3_weight_sum= [],[],[]
min_fc1_weight_sum, min_fc2_weight_sum, min_fc3_weight_sum= [],[],[]
# Iterate through model_paths1
for path in model_paths3:
    # Execute stringtotensor() function on each path
    max_fc1_weight_tensor = stringtotensor(path, "model_epoch20 fc1.weight")
    # Stack the resulting tensor in max_fc1_weight_sum
    max_fc1_weight_sum.append(max_fc1_weight_tensor)
    
    max_fc2_weight_tensor = stringtotensor(path, "model_epoch20 fc2.weight")
    max_fc2_weight_sum.append(max_fc2_weight_tensor)
    
    max_fc3_weight_tensor = stringtotensor(path, "model_epoch20 fc3.weight")
    max_fc3_weight_sum.append(max_fc3_weight_tensor)
# Convert max_fc1_weight_sum to a tensor
max_fc1_weight_sum = torch.stack(max_fc1_weight_sum).flatten()
max_fc2_weight_sum = torch.stack(max_fc2_weight_sum).flatten()
max_fc3_weight_sum = torch.stack(max_fc3_weight_sum).flatten()

for path in model_paths2:
    # Execute stringtotensor() function on each path
    mid_fc1_weight_tensor = stringtotensor(path, "model_epoch20 fc1.weight")
    # Stack the resulting tensor in max_fc1_weight_sum
    mid_fc1_weight_sum.append(mid_fc1_weight_tensor)
    
    mid_fc2_weight_tensor = stringtotensor(path, "model_epoch20 fc2.weight")
    mid_fc2_weight_sum.append(mid_fc2_weight_tensor)
    
    mid_fc3_weight_tensor = stringtotensor(path, "model_epoch20 fc3.weight")
    mid_fc3_weight_sum.append(mid_fc3_weight_tensor)
# Convert max_fc1_weight_sum to a tensor
mid_fc1_weight_sum = torch.stack(mid_fc1_weight_sum).flatten()
mid_fc2_weight_sum = torch.stack(mid_fc2_weight_sum).flatten()
mid_fc3_weight_sum = torch.stack(mid_fc3_weight_sum).flatten()

for path in model_paths1:
    # Execute stringtotensor() function on each path
    min_fc1_weight_tensor = stringtotensor(path, "model_epoch20 fc1.weight")
    # Stack the resulting tensor in max_fc1_weight_sum
    min_fc1_weight_sum.append(min_fc1_weight_tensor)
    
    min_fc2_weight_tensor = stringtotensor(path, "model_epoch20 fc2.weight")
    min_fc2_weight_sum.append(min_fc2_weight_tensor)
    
    min_fc3_weight_tensor = stringtotensor(path, "model_epoch20 fc3.weight")
    min_fc3_weight_sum.append(min_fc3_weight_tensor)
# Convert max_fc1_weight_sum to a tensor
min_fc1_weight_sum = torch.stack(min_fc1_weight_sum).flatten()
min_fc2_weight_sum = torch.stack(min_fc2_weight_sum).flatten()
min_fc3_weight_sum = torch.stack(min_fc3_weight_sum).flatten()


painthistoverlap(max_fc1_weight_sum,mid_fc1_weight_sum,min_fc1_weight_sum,1)
painthistoverlap(max_fc2_weight_sum,mid_fc2_weight_sum,min_fc2_weight_sum,2)
painthistoverlap(max_fc3_weight_sum,mid_fc3_weight_sum,min_fc3_weight_sum,3)
all_max_weight=tensoradd2(max_fc1_weight_sum,max_fc2_weight_sum,max_fc3_weight_sum)
all_mid_weight=tensoradd2(mid_fc1_weight_sum,mid_fc2_weight_sum,mid_fc3_weight_sum)
all_min_weight=tensoradd2(min_fc1_weight_sum,min_fc2_weight_sum,min_fc3_weight_sum)
painthistoverlap(all_max_weight,all_mid_weight,all_min_weight,4)
plt.tight_layout()
plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-fashionmnist-plot/whole_gradients_distribution_overlap.png")
plt.show()