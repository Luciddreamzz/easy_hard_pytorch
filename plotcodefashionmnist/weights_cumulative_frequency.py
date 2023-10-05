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
from scipy import stats
import seaborn as sns 
import matplotlib as mpl 
from torch import tensor
torch.set_printoptions(threshold=np.inf) #Display all data for complex matrix 显示所有省略号的玩意
#---------------------------------把字符串转换成张量 let string transform to tensor----------------------------------------------------
def stringtotensor(path,name):
    #fig, ax = plt.subplots()
    f=open(path,'rb')
    dod=torch.load(f)
    torch.set_printoptions(threshold=np.inf) #Display all data for complex matrix 显示所有省略号的玩意
    tensor=dod[name]
    return tensor
def freedman_diaconis_rule(data):
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    bin_width = 2 * iqr / len(data) ** (1/3)
    return bin_width
def getbins(fc_weight):
    bins=np.arange(min(fc_weight).item(), max(fc_weight).item() +0.01, 0.01)
    return bins
#---------------------------------把所有张量写成一个 add all tensor into one tensor----------------------------------------------------
def tensoradd(tensor1,tensor2,tensor3,tensor4,tensor5,tensor6,tensor7,tensor8,tensor9,tensor10):
# concatenate the two tensors along dimension 0
    concatenated_tensor = torch.cat((tensor1,tensor2,tensor3,tensor4,tensor5,tensor6,tensor7,tensor8,tensor9,tensor10), dim=0).flatten()
    return concatenated_tensor

#---------------------------------把所有张量写成一个 add all tensor into one tensor----------------------------------------------------
def tensoradd2(tensor1,tensor2,tensor3):
# concatenate the two tensors along dimension 0
    concatenated_tensor = torch.cat((tensor1,tensor2,tensor3), dim=0).flatten()
    return concatenated_tensor


def painthistcumulative(fc_weight_max,fc_weight_mid,fc_weight_min):

    min_cumfreq = stats.cumfreq(fc_weight_min,numbins=100)
    x_min=min_cumfreq.lowerlimit+np.linspace(0,min_cumfreq.binsize*min_cumfreq.cumcount.size,min_cumfreq.cumcount.size)
    mid_cumfreq = stats.cumfreq(fc_weight_mid,numbins=100)
    x_mid=mid_cumfreq.lowerlimit+np.linspace(0,mid_cumfreq.binsize*mid_cumfreq.cumcount.size,mid_cumfreq.cumcount.size)
    max_cumfreq = stats.cumfreq(fc_weight_max,numbins=100)
    x_max=max_cumfreq.lowerlimit+np.linspace(0,max_cumfreq.binsize*max_cumfreq.cumcount.size,max_cumfreq.cumcount.size)
    fig = plt.figure(figsize=(6, 9))
    ax1 = fig.add_subplot(3, 2, 1)
    ax2 = fig.add_subplot(3, 2, 2)
    ax3 = fig.add_subplot(3, 2, 3)
    ax4 = fig.add_subplot(3, 2, 4)
    ax5 = fig.add_subplot(3, 2, 5)
    ax6 = fig.add_subplot(3, 2, 6)

    ax1.hist(fc_weight_max, bins=getbins(fc_weight_max),color="gold")
    ax1.set_title('Weight histogram of max accuracy', fontsize=7)
    ax1.set_xlabel('Weights',fontsize=6)
    ax1.set_ylabel('frequency', fontsize=6)
    ax1.tick_params(axis='both', which='major', labelsize=4)
    ax1.set_xlim(-1.5,1.5)
    
    ax2.bar(x_max, max_cumfreq.cumcount,width=0.02,color="gold")
    ax2.set_title('Weight Cumulative histogram of max accuracy', fontsize=7)
    ax2.set_xlabel('Weights',fontsize=6)
    ax2.set_ylabel('Cumulative frequency', fontsize=6)
    ax2.tick_params(axis='both', which='major', labelsize=4)
    ax2.set_xlim(-1,2)
    
    ax3.hist(fc_weight_mid, bins=getbins(fc_weight_mid),color="deepskyblue")
    ax3.set_title('Weight histogram of mid accuracy', fontsize=7)
    ax3.set_xlabel('Weights',fontsize=6)
    ax3.set_ylabel('frequency', fontsize=6)
    ax3.tick_params(axis='both', which='major', labelsize=4)
    ax3.set_xlim(-1.5,1.5)
    
    ax4.bar(x_mid, mid_cumfreq.cumcount,width=0.02,color="deepskyblue")
    ax4.set_title('Weight Cumulative histogram of mid accuracy', fontsize=7)
    ax4.set_xlabel('Weights',fontsize=6)
    ax4.set_ylabel('Cumulative frequency', fontsize=6)
    ax4.tick_params(axis='both', which='major', labelsize=4)
    ax4.set_xlim(-1,2)

    
    ax5.hist(fc_weight_min, bins=getbins(fc_weight_min),color="salmon")
    ax5.set_title('Weight histogram of min accuracy', fontsize=7)
    ax5.set_xlabel('Weights',fontsize=6)
    ax5.set_ylabel('frequency', fontsize=6)
    ax5.tick_params(axis='both', which='major', labelsize=4)
    ax5.set_xlim(-0.05,0.05)

    
    ax6.bar(x_min, min_cumfreq.cumcount,width=0.02,color="salmon")
    ax6.set_title('Weight Cumulative histogram of min accuracy', fontsize=7)
    ax6.set_xlabel('Weights',fontsize=6)
    ax6.set_ylabel('Cumulative frequency', fontsize=6)
    ax6.tick_params(axis='both', which='major', labelsize=4)
    ax6.set_xlim(-0.05,0.05)
    # ax4.set_ylim(0,100000)

    plt.tight_layout()
    plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-fashionmnist-plot/weights_cumulative.png", dpi=300)
    plt.show()



#---------------------------------main function---------------------------------------------------- 
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

#---------------------------------max fc1 weight----------------------------------------------------
max_fc1_weight_sum,max_fc2_weight_sum,max_fc3_weight_sum = [],[],[]
mid_fc1_weight_sum,mid_fc2_weight_sum,mid_fc3_weight_sum= [],[],[]
min_fc1_weight_sum, min_fc2_weight_sum, min_fc3_weight_sum= [],[],[]
# Iterate through model_paths1
for path in model_paths3:
    # Execute stringtotensor() function on each path
    max_fc1_weight_tensor = stringtotensor(path, "fc1.weight")
    # Stack the resulting tensor in max_fc1_weight_sum
    max_fc1_weight_sum.append(max_fc1_weight_tensor)
    
    max_fc2_weight_tensor = stringtotensor(path, "fc2.weight")
    max_fc2_weight_sum.append(max_fc2_weight_tensor)
    
    max_fc3_weight_tensor = stringtotensor(path, "fc3.weight")
    max_fc3_weight_sum.append(max_fc3_weight_tensor)
# Convert max_fc1_weight_sum to a tensor
max_fc1_weight_sum = torch.stack(max_fc1_weight_sum).flatten()
max_fc2_weight_sum = torch.stack(max_fc2_weight_sum).flatten()
max_fc3_weight_sum = torch.stack(max_fc3_weight_sum).flatten()

for path in model_paths2:
    # Execute stringtotensor() function on each path
    mid_fc1_weight_tensor = stringtotensor(path, "fc1.weight")
    # Stack the resulting tensor in max_fc1_weight_sum
    mid_fc1_weight_sum.append(mid_fc1_weight_tensor)
    
    mid_fc2_weight_tensor = stringtotensor(path, "fc2.weight")
    mid_fc2_weight_sum.append(mid_fc2_weight_tensor)
    
    mid_fc3_weight_tensor = stringtotensor(path, "fc3.weight")
    mid_fc3_weight_sum.append(mid_fc3_weight_tensor)
# Convert max_fc1_weight_sum to a tensor
mid_fc1_weight_sum = torch.stack(mid_fc1_weight_sum).flatten()
mid_fc2_weight_sum = torch.stack(mid_fc2_weight_sum).flatten()
mid_fc3_weight_sum = torch.stack(mid_fc3_weight_sum).flatten()

for path in model_paths1:
    # Execute stringtotensor() function on each path
    min_fc1_weight_tensor = stringtotensor(path, "fc1.weight")
    # Stack the resulting tensor in max_fc1_weight_sum
    min_fc1_weight_sum.append(min_fc1_weight_tensor)
    
    min_fc2_weight_tensor = stringtotensor(path, "fc2.weight")
    min_fc2_weight_sum.append(min_fc2_weight_tensor)
    
    min_fc3_weight_tensor = stringtotensor(path, "fc3.weight")
    min_fc3_weight_sum.append(min_fc3_weight_tensor)
# Convert max_fc1_weight_sum to a tensor
min_fc1_weight_sum = torch.stack(min_fc1_weight_sum).flatten()
min_fc2_weight_sum = torch.stack(min_fc2_weight_sum).flatten()
min_fc3_weight_sum = torch.stack(min_fc3_weight_sum).flatten()


all_max_weight=tensoradd2(max_fc1_weight_sum,max_fc2_weight_sum,max_fc3_weight_sum)
all_mid_weight=tensoradd2(mid_fc1_weight_sum,mid_fc2_weight_sum,mid_fc3_weight_sum)
all_min_weight=tensoradd2(min_fc1_weight_sum,min_fc2_weight_sum,min_fc3_weight_sum)

painthistcumulative(all_max_weight,all_mid_weight,all_min_weight)
