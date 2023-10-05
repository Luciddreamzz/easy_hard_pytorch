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

#---------------------------------Scott_rule规则用于确定直方图中的条柱宽度 The Scott_rule is used to determine the width of the bars in the histogram----------------------------------------------------
def Scott_rule(data):
    bin_width = 3.5 * np.std(data.numpy()) / len(data) ** (1/3)
    print(bin_width)
    return bin_width
#---------------------------------把所有张量写成一个 stack all tensor into one tensor----------------------------------------------------
def getbins(fc_weight):
    bins=np.arange(min(fc_weight).item(), max(fc_weight).item() + Scott_rule(fc_weight), Scott_rule(fc_weight))
    return bins

#---------------------------------把所有张量写成一个 stack all tensor into one tensor----------------------------------------------------
def getbins1(fc_weight):
    bins=np.arange(min(fc_weight).item(), max(fc_weight).item() + 0.1, 0.1)
    return bins
#---------------------------------把所有张量写成一个 add all tensor into one tensor----------------------------------------------------
def tensoradd2(tensor1,tensor2,tensor3):
# concatenate the two tensors along dimension 0
    concatenated_tensor = torch.cat((tensor1,tensor2,tensor3), dim=0).flatten()
    return concatenated_tensor
#---------------------------------画图----------------------------------------------------
def painthistoverlap(fc_weight_max,flag):
    
    plt.subplot(1,4,flag)
    if flag==4:
        sns.distplot(fc_weight_max,bins=getbins1(fc_weight_max),kde_kws={"color":"blueviolet", "lw":1 ,"linestyle":"-","label":"max accuracy"}, hist_kws={ "color": "blueviolet" }) #lightcoral mediumturquoise
    else:
        sns.distplot(fc_weight_max,bins=getbins(fc_weight_max),kde_kws={"color":"blueviolet", "lw":1 ,"linestyle":"-","label":"max accuracy"}, hist_kws={ "color": "blueviolet" }) #lightcoral mediumturquoise
    
    plt.xticks(fontproperties = 'Times New Roman', size = 6)
    plt.yticks(fontproperties = 'Times New Roman', size = 6)
    plt.title("",fontsize=8)
    plt.xlabel("gradients",fontsize=6)
    plt.ylabel("Frequency",fontsize=6)
    plt.legend(loc='best',fontsize=6)


#---------------------------------main function---------------------------------------------------- 
# Initialize an empty list to store the tensors
tensor_list1 = []
tensor_list2 = []
tensor_list3 = []
# Loop through each path and call the stringtotensor() function to get the tensor
for i in range(10):
    for j in range(10):
        path1 = f"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-{i+1}/train_result{j+1}/gradient_dic.pkl"

        # Get the tensor from the path
        tensor1 = stringtotensor(path1, name="model_epoch20 fc1.weight").flatten()
        tensor2 = stringtotensor(path1, name="model_epoch20 fc2.weight").flatten()
        tensor3 = stringtotensor(path1, name="model_epoch20 fc3.weight").flatten()
        # Append the tensor to the list
        tensor_list1.append(tensor1)
        tensor_list2.append(tensor2)
        tensor_list3.append(tensor3)

# Stack the tensors in the list along the specified dimension (in this case, 0)
fc1_gradient = torch.cat(tensor_list1, dim=0).flatten()
fc2_gradient = torch.cat(tensor_list2, dim=0).flatten()
fc3_gradient = torch.cat(tensor_list3, dim=0).flatten()
all_gradient= torch.cat((fc1_gradient,fc2_gradient, fc3_gradient), dim=0).flatten()

plt.figure(figsize=(15,3),dpi=300)
painthistoverlap(fc1_gradient,1)
painthistoverlap(fc2_gradient,2)
painthistoverlap(fc3_gradient,3)
painthistoverlap(all_gradient,4)
plt.tight_layout()
#plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/whole_gradients_distribution_overlap.png")
plt.show()