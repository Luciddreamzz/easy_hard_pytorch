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


def painthistcumulative(fc_weight_max, fc_weight_highermid,fc_weight_lowermid, fc_weight_min):

    min_cumfreq = stats.cumfreq(fc_weight_min,numbins=100)
    x_min=min_cumfreq.lowerlimit+np.linspace(0,min_cumfreq.binsize*min_cumfreq.cumcount.size,min_cumfreq.cumcount.size)
    highermid_cumfreq = stats.cumfreq(fc_weight_highermid,numbins=100)
    x_highermid=highermid_cumfreq.lowerlimit+np.linspace(0,highermid_cumfreq.binsize*highermid_cumfreq.cumcount.size,highermid_cumfreq.cumcount.size)
    lowermid_cumfreq = stats.cumfreq(fc_weight_lowermid,numbins=100)
    x_lowermid=lowermid_cumfreq.lowerlimit+np.linspace(0,lowermid_cumfreq.binsize*lowermid_cumfreq.cumcount.size,lowermid_cumfreq.cumcount.size)
    max_cumfreq = stats.cumfreq(fc_weight_max,numbins=100)
    x_max=max_cumfreq.lowerlimit+np.linspace(0,max_cumfreq.binsize*max_cumfreq.cumcount.size,max_cumfreq.cumcount.size)
    fig = plt.figure(figsize=(6, 12))
    ax1 = fig.add_subplot(4, 2, 1)
    ax2 = fig.add_subplot(4, 2, 2)
    ax3 = fig.add_subplot(4, 2, 3)
    ax4 = fig.add_subplot(4, 2, 4)
    ax5 = fig.add_subplot(4, 2, 5)
    ax6 = fig.add_subplot(4, 2, 6)
    ax7 = fig.add_subplot(4, 2, 7)
    ax8 = fig.add_subplot(4, 2, 8)
    #ax1.hist(fc_weight_min, bins=25, color="green")    
    #ax1.set_title('Histogram')
    ax1.hist(fc_weight_max, bins=getbins(fc_weight_max),color="gold")
    ax1.set_title('Gradients histogram of max accuracy', fontsize=7)
    ax1.set_xlabel('Gradients',fontsize=6)
    ax1.set_ylabel('frequency', fontsize=6)
    ax1.tick_params(axis='both', which='major', labelsize=4)
    ax1.set_xlim(-0.1,0.1)
    # ax1.set_ylim(0,25000)
    
    ax2.bar(x_max, max_cumfreq.cumcount,width=0.02,color="gold")
    ax2.set_title('Gradients Cumulative histogram of max accuracy', fontsize=7)
    ax2.set_xlabel('Gradients',fontsize=6)
    ax2.set_ylabel('Cumulative frequency', fontsize=6)
    ax2.tick_params(axis='both', which='major', labelsize=4)
    ax2.set_xlim(-0.5,1.5)
    # ax2.set_ylim(0,100000)
    
    ax3.hist(fc_weight_highermid, bins=getbins(fc_weight_highermid),color="salmon")
    ax3.set_title('Gradients histogram of highermid accuracy', fontsize=7)
    ax3.set_xlabel('Gradients',fontsize=6)
    ax3.set_ylabel('frequency', fontsize=6)
    ax3.tick_params(axis='both', which='major', labelsize=4)
    ax3.set_xlim(-0.1,0.1)
    ax3.set_ylim(0,40000)
    
    ax4.bar(x_highermid, highermid_cumfreq.cumcount,width=0.02,color="salmon")
    ax4.set_title('Gradients Cumulative histogram of highermid accuracy', fontsize=7)
    ax4.set_xlabel('Gradients',fontsize=6)
    ax4.set_ylabel('Cumulative frequency', fontsize=6)
    ax4.tick_params(axis='both', which='major', labelsize=4)
    ax4.set_xlim(-1,5)
    # ax4.set_ylim(0,100000)
    
    ax5.hist(fc_weight_lowermid, bins=getbins(fc_weight_lowermid),color="lightgreen")
    ax5.set_title('Gradients histogram of lowermid accuracy', fontsize=7)
    ax5.set_xlabel('Gradients',fontsize=6)
    ax5.set_ylabel('frequency', fontsize=6)
    ax5.tick_params(axis='both', which='major', labelsize=4)
    ax5.set_xlim(-0.05,0.05)
    ax5.set_ylim(0,100000)
    
    ax6.bar(x_lowermid, lowermid_cumfreq.cumcount,width=0.02,color="lightgreen")
    ax6.set_title('Gradients Cumulative histogram of lowermid accuracy', fontsize=7)
    ax6.set_xlabel('Gradients',fontsize=6)
    ax6.set_ylabel('Cumulative frequency', fontsize=6)
    ax6.tick_params(axis='both', which='major', labelsize=4)
    ax6.set_xlim(-1,7)
    # ax6.set_ylim(0,100000)
    
    ax7.hist(fc_weight_min, bins=getbins(fc_weight_min),color="lightskyblue")
    ax7.set_title('Gradients histogram of min accuracy', fontsize=7)
    ax7.set_xlabel('Gradients',fontsize=6)
    ax7.set_ylabel('frequency', fontsize=6)
    ax7.tick_params(axis='both', which='major', labelsize=4)
    # ax7.set_xlim(-0.05,0.05)
    # ax7.set_ylim(0,40000)
    
    ax8.bar(x_min, min_cumfreq.cumcount,width=0.02,color="lightskyblue")
    ax8.set_title('Gradients Cumulative histogram of min accuracy', fontsize=7)
    ax8.set_xlabel('Gradients',fontsize=6)
    ax8.set_ylabel('Cumulative frequency', fontsize=6)
    ax8.tick_params(axis='both', which='major', labelsize=4)
    # ax8.set_xlim(-0.05,0.15)
    # ax8.set_ylim(0,130000)
    
    plt.tight_layout()
    plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-mnist-plot/gradient_cumulative.png", dpi=300)
    plt.show()




#---------------------------------main function---------------------------------------------------- 
model_paths = []    
for i in range(1, 101):
    for j in range(1, 11):
        path = "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-mnist/b100l5lr0001-{}/train_result{}/gradient_dic.pkl".format(i, j)
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
        # Check if the test accuracy is less than 0.2
        if test_acc < 0.2:
            model_paths1.append(path)
        # Check if the test accuracy is in the range [0.2, 0.5)
        if 0.2 <= test_acc < 0.5:
            model_paths2.append(path)
        # compute the difference between test_acc and 0.65
        diff1 = abs(test_acc - 0.65)
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
max_fc1_weight_sum,max_fc2_weight_sum,max_fc3_weight_sum = [],[],[]
highermid_fc1_weight_sum,highermid_fc2_weight_sum,highermid_fc3_weight_sum= [],[],[]
lowermid_fc1_weight_sum,lowermid_fc2_weight_sum,lowermid_fc3_weight_sum = [],[],[]
min_fc1_weight_sum, min_fc2_weight_sum, min_fc3_weight_sum= [],[],[]
# Iterate through model_paths1
for path in model_paths4:
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

for path in model_paths3:
    # Execute stringtotensor() function on each path
    highermid_fc1_weight_tensor = stringtotensor(path, "model_epoch20 fc1.weight")
    # Stack the resulting tensor in max_fc1_weight_sum
    highermid_fc1_weight_sum.append(highermid_fc1_weight_tensor)
    
    highermid_fc2_weight_tensor = stringtotensor(path, "model_epoch20 fc2.weight")
    highermid_fc2_weight_sum.append(highermid_fc2_weight_tensor)
    
    highermid_fc3_weight_tensor = stringtotensor(path, "model_epoch20 fc3.weight")
    highermid_fc3_weight_sum.append(highermid_fc3_weight_tensor)
# Convert max_fc1_weight_sum to a tensor
highermid_fc1_weight_sum = torch.stack(highermid_fc1_weight_sum).flatten()
highermid_fc2_weight_sum = torch.stack(highermid_fc2_weight_sum).flatten()
highermid_fc3_weight_sum = torch.stack(highermid_fc3_weight_sum).flatten()

for path in model_paths2:
    # Execute stringtotensor() function on each path
    lowermid_fc1_weight_tensor = stringtotensor(path, "model_epoch20 fc1.weight")
    # Stack the resulting tensor in max_fc1_weight_sum
    lowermid_fc1_weight_sum.append(lowermid_fc1_weight_tensor)
    
    lowermid_fc2_weight_tensor = stringtotensor(path, "model_epoch20 fc2.weight")
    lowermid_fc2_weight_sum.append(lowermid_fc2_weight_tensor)
    
    lowermid_fc3_weight_tensor = stringtotensor(path, "model_epoch20 fc3.weight")
    lowermid_fc3_weight_sum.append(lowermid_fc3_weight_tensor)
# Convert max_fc1_weight_sum to a tensor
lowermid_fc1_weight_sum = torch.stack(lowermid_fc1_weight_sum).flatten()
lowermid_fc2_weight_sum = torch.stack(lowermid_fc2_weight_sum).flatten()
lowermid_fc3_weight_sum = torch.stack(lowermid_fc3_weight_sum).flatten()

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


all_max_weight=tensoradd2(max_fc1_weight_sum,max_fc2_weight_sum,max_fc3_weight_sum)
all_highermid_weight=tensoradd2(highermid_fc1_weight_sum,highermid_fc2_weight_sum,highermid_fc3_weight_sum)
all_lowermid_weight=tensoradd2(lowermid_fc1_weight_sum,lowermid_fc2_weight_sum,lowermid_fc3_weight_sum)
all_min_weight=tensoradd2(min_fc1_weight_sum,min_fc2_weight_sum,min_fc3_weight_sum)

painthistcumulative(all_max_weight,all_highermid_weight,all_lowermid_weight,all_min_weight)
