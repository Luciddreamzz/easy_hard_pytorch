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
#---------------------------------提取文件中的权重 get weight----------------------------------------------------
def stringtotensor(path,name):
    #fig, ax = plt.subplots()
    f=open(path,'rb')
    dod=torch.load(f)
    torch.set_printoptions(threshold=np.inf) #Display all data for complex matrix 显示所有省略号的玩意
    tensor=dod[name]
    return tensor

def csvfilesave(csvfile_path,list_name):
    csvFile = open(csvfile_path, "a+",)
    try:
        writer = csv.writer(csvFile)
        writer.writerow(list_name)
    finally:
        csvFile.close() 
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
#---------------------------------Scott rule规则用于确定直方图中的条柱宽度 The Scott rule is used to determine the width of the bars in the histogram----------------------------------------------------
def Scott_rule(data):
    bin_width = 3.5 * np.std(data.numpy()) / len(data) ** (1/3)
    print(bin_width)
    return bin_width
#---------------------------------Freedman-Diaconis规则用于确定直方图中的条柱宽度 The Freedman-Diaconis rule is used to determine the width of the bars in the histogram----------------------------------------------------
def freedman_diaconis_rule(data):
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    bin_width = 2 * iqr / len(data) ** (1/3)
    return bin_width
#---------------------------------sturges_rule规则用于确定直方图中的条柱宽度 The sturges_rule is used to determine the width of the bars in the histogram----------------------------------------------------
def sturges_rule(data):
    """Compute the number of bins using Sturges' rule."""
    n = len(data)
    k = int(np.ceil(np.log2(n+1)))
    return k
#---------------------------------把所有张量写成一个 stack all tensor into one tensor----------------------------------------------------
#def getbins(fc_weight):
    #bins=np.arange(min(fc_weight).item(), max(fc_weight).item() + 0.1, 0.1)
    ##return bins

#def painthistoverlap(fc_weight_max,fc_weight_mid,fc_weight_min,figure_name):
    
    #sns.distplot(fc_weight_max,bins=getbins(fc_weight_max),kde_kws={"color":"blueviolet", "lw":1 ,"linestyle":"-","label":"max accuracy"}, hist_kws={ "color": "blueviolet" }) #lightcoral mediumturquoise
    #sns.distplot(fc_weight_mid,bins=getbins(fc_weight_mid),kde_kws={"color":"deepskyblue", "lw":1 ,"linestyle":"-","label":"mid accuracy"}, hist_kws={ "color": "deepskyblue" })
    #sns.distplot(fc_weight_min,bins=getbins(fc_weight_min),kde_kws={"color":"lightcoral", "lw":1 ,"linestyle":"-","label":"min accuracy"}, hist_kws={ "color": "lightcoral" })   
    #plt.xticks(fontproperties = 'Times New Roman', size = 6)
    #plt.yticks(fontproperties = 'Times New Roman', size = 6)
    #plt.title(figure_name,fontsize=8)
    #plt.xlabel("weights",fontsize=6)
    #plt.ylabel("Frequency",fontsize=6)
    #plt.legend(loc='best',fontsize=8)
def mean_std(network):
    mean=np.mean(network.tolist())
    std=np.std(network.tolist())
    return mean,std

#---------------------------------main function---------------------------------------------------- 

#plt.figure(figsize=(9,9),dpi=300)
#---------------------------------max fc1 weight----------------------------------------------------
# create your two tensors
max_fc1_weight_tensor1 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result5/cnnmodel.pkl","fc1.weight")
max_fc1_weight_tensor2 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result6/cnnmodel.pkl","fc1.weight")
max_fc1_weight_tensor3 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result5/cnnmodel.pkl","fc1.weight")
max_fc1_weight_tensor4 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result5/cnnmodel.pkl","fc1.weight")
max_fc1_weight_tensor5 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result6/cnnmodel.pkl","fc1.weight")
max_fc1_weight_tensor6 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result8/cnnmodel.pkl","fc1.weight")
max_fc1_weight_tensor7 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result10/cnnmodel.pkl","fc1.weight")
max_fc1_weight_tensor8 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result2/cnnmodel.pkl","fc1.weight")
max_fc1_weight_tensor9 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result2/cnnmodel.pkl","fc1.weight")
max_fc1_weight_tensor10 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result10/cnnmodel.pkl","fc1.weight")
max_fc1_weight_sum=tensoradd(max_fc1_weight_tensor1,max_fc1_weight_tensor2,max_fc1_weight_tensor3,max_fc1_weight_tensor4,max_fc1_weight_tensor5,
          max_fc1_weight_tensor6,max_fc1_weight_tensor7,max_fc1_weight_tensor8,max_fc1_weight_tensor9,max_fc1_weight_tensor10)

#---------------------------------mid fc1 weight----------------------------------------------------
mid_fc1_weight_tensor1 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result10/cnnmodel.pkl","fc1.weight")
mid_fc1_weight_tensor2 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result9/cnnmodel.pkl","fc1.weight")
mid_fc1_weight_tensor3 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result4/cnnmodel.pkl","fc1.weight")
mid_fc1_weight_tensor4 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result5/cnnmodel.pkl","fc1.weight")
mid_fc1_weight_tensor5 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result8/cnnmodel.pkl","fc1.weight")
mid_fc1_weight_tensor6 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result10/cnnmodel.pkl","fc1.weight")
mid_fc1_weight_tensor7 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result8/cnnmodel.pkl","fc1.weight")
mid_fc1_weight_tensor8 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result9/cnnmodel.pkl","fc1.weight")
mid_fc1_weight_tensor9 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result1/cnnmodel.pkl","fc1.weight")
mid_fc1_weight_tensor10 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result10/cnnmodel.pkl","fc1.weight")
mid_fc1_weight_sum=tensoradd(mid_fc1_weight_tensor1,mid_fc1_weight_tensor2,mid_fc1_weight_tensor3,mid_fc1_weight_tensor4,mid_fc1_weight_tensor5,
          mid_fc1_weight_tensor6,mid_fc1_weight_tensor7,mid_fc1_weight_tensor8,mid_fc1_weight_tensor9,mid_fc1_weight_tensor10)

#---------------------------------min fc1 weight----------------------------------------------------
min_fc1_weight_tensor1 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result3/cnnmodel.pkl","fc1.weight")
min_fc1_weight_tensor2 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result3/cnnmodel.pkl","fc1.weight")
min_fc1_weight_tensor3 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result8/cnnmodel.pkl","fc1.weight")
min_fc1_weight_tensor4 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result9/cnnmodel.pkl","fc1.weight")
min_fc1_weight_tensor5 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result10/cnnmodel.pkl","fc1.weight")
min_fc1_weight_tensor6 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result1/cnnmodel.pkl","fc1.weight")
min_fc1_weight_tensor7 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result6/cnnmodel.pkl","fc1.weight")
min_fc1_weight_tensor8 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result6/cnnmodel.pkl","fc1.weight")
min_fc1_weight_tensor9 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-8/train_result2/cnnmodel.pkl","fc1.weight")
min_fc1_weight_tensor10 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result3/cnnmodel.pkl","fc1.weight")
min_fc1_weight_sum=tensoradd(min_fc1_weight_tensor1,min_fc1_weight_tensor2,min_fc1_weight_tensor3,min_fc1_weight_tensor4,min_fc1_weight_tensor5,
          min_fc1_weight_tensor6,min_fc1_weight_tensor7,min_fc1_weight_tensor8,min_fc1_weight_tensor9,min_fc1_weight_tensor10)

#---------------------------------max fc2 weight----------------------------------------------------
# create your two tensors
max_fc2_weight_tensor1 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result5/cnnmodel.pkl","fc2.weight")
max_fc2_weight_tensor2 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result6/cnnmodel.pkl","fc2.weight")
max_fc2_weight_tensor3 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result5/cnnmodel.pkl","fc2.weight")
max_fc2_weight_tensor4 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result5/cnnmodel.pkl","fc2.weight")
max_fc2_weight_tensor5 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result6/cnnmodel.pkl","fc2.weight")
max_fc2_weight_tensor6 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result8/cnnmodel.pkl","fc2.weight")
max_fc2_weight_tensor7 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result10/cnnmodel.pkl","fc2.weight")
max_fc2_weight_tensor8 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result2/cnnmodel.pkl","fc2.weight")
max_fc2_weight_tensor9 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result2/cnnmodel.pkl","fc2.weight")
max_fc2_weight_tensor10 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result10/cnnmodel.pkl","fc2.weight")
max_fc2_weight_sum=tensoradd(max_fc2_weight_tensor1,max_fc2_weight_tensor2,max_fc2_weight_tensor3,max_fc2_weight_tensor4,max_fc2_weight_tensor5,
          max_fc2_weight_tensor6,max_fc2_weight_tensor7,max_fc2_weight_tensor8,max_fc2_weight_tensor9,max_fc2_weight_tensor10)

#---------------------------------mid fc2 weight----------------------------------------------------
mid_fc2_weight_tensor1 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result10/cnnmodel.pkl","fc2.weight")
mid_fc2_weight_tensor2 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result9/cnnmodel.pkl","fc2.weight")
mid_fc2_weight_tensor3 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result4/cnnmodel.pkl","fc2.weight")
mid_fc2_weight_tensor4 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result5/cnnmodel.pkl","fc2.weight")
mid_fc2_weight_tensor5 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result8/cnnmodel.pkl","fc2.weight")
mid_fc2_weight_tensor6 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result10/cnnmodel.pkl","fc2.weight")
mid_fc2_weight_tensor7 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result8/cnnmodel.pkl","fc2.weight")
mid_fc2_weight_tensor8 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result9/cnnmodel.pkl","fc2.weight")
mid_fc2_weight_tensor9 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result1/cnnmodel.pkl","fc2.weight")
mid_fc2_weight_tensor10 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result10/cnnmodel.pkl","fc2.weight")
mid_fc2_weight_sum=tensoradd(mid_fc2_weight_tensor1,mid_fc2_weight_tensor2,mid_fc2_weight_tensor3,mid_fc2_weight_tensor4,mid_fc2_weight_tensor5,
          mid_fc2_weight_tensor6,mid_fc2_weight_tensor7,mid_fc2_weight_tensor8,mid_fc2_weight_tensor9,mid_fc2_weight_tensor10)

#---------------------------------min fc2 weight----------------------------------------------------
min_fc2_weight_tensor1 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result3/cnnmodel.pkl","fc2.weight")
min_fc2_weight_tensor2 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result3/cnnmodel.pkl","fc2.weight")
min_fc2_weight_tensor3 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result8/cnnmodel.pkl","fc2.weight")
min_fc2_weight_tensor4 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result9/cnnmodel.pkl","fc2.weight")
min_fc2_weight_tensor5 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result10/cnnmodel.pkl","fc2.weight")
min_fc2_weight_tensor6 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result1/cnnmodel.pkl","fc2.weight")
min_fc2_weight_tensor7 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result6/cnnmodel.pkl","fc2.weight")
min_fc2_weight_tensor8 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result6/cnnmodel.pkl","fc2.weight")
min_fc2_weight_tensor9 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-8/train_result2/cnnmodel.pkl","fc2.weight")
min_fc2_weight_tensor10 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result3/cnnmodel.pkl","fc2.weight")
min_fc2_weight_sum=tensoradd(min_fc2_weight_tensor1,min_fc2_weight_tensor2,min_fc2_weight_tensor3,min_fc2_weight_tensor4,min_fc2_weight_tensor5,
          min_fc2_weight_tensor6,min_fc2_weight_tensor7,min_fc2_weight_tensor8,min_fc2_weight_tensor9,min_fc2_weight_tensor10)

#---------------------------------max fc3 weight----------------------------------------------------
# create your two tensors
max_fc3_weight_tensor1 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result5/cnnmodel.pkl","fc3.weight")
max_fc3_weight_tensor2 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result6/cnnmodel.pkl","fc3.weight")
max_fc3_weight_tensor3 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result5/cnnmodel.pkl","fc3.weight")
max_fc3_weight_tensor4 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result5/cnnmodel.pkl","fc3.weight")
max_fc3_weight_tensor5 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result6/cnnmodel.pkl","fc3.weight")
max_fc3_weight_tensor6 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result8/cnnmodel.pkl","fc3.weight")
max_fc3_weight_tensor7 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result10/cnnmodel.pkl","fc3.weight")
max_fc3_weight_tensor8 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result2/cnnmodel.pkl","fc3.weight")
max_fc3_weight_tensor9 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result2/cnnmodel.pkl","fc3.weight")
max_fc3_weight_tensor10 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result10/cnnmodel.pkl","fc3.weight")
max_fc3_weight_sum=tensoradd(max_fc3_weight_tensor1,max_fc3_weight_tensor2,max_fc3_weight_tensor3,max_fc3_weight_tensor4,max_fc3_weight_tensor5,
          max_fc3_weight_tensor6,max_fc3_weight_tensor7,max_fc3_weight_tensor8,max_fc3_weight_tensor9,max_fc3_weight_tensor10)

#---------------------------------mid fc3 weight----------------------------------------------------
mid_fc3_weight_tensor1 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result10/cnnmodel.pkl","fc3.weight")
mid_fc3_weight_tensor2 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result9/cnnmodel.pkl","fc3.weight")
mid_fc3_weight_tensor3 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result4/cnnmodel.pkl","fc3.weight")
mid_fc3_weight_tensor4 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result5/cnnmodel.pkl","fc3.weight")
mid_fc3_weight_tensor5 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result8/cnnmodel.pkl","fc3.weight")
mid_fc3_weight_tensor6 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result10/cnnmodel.pkl","fc3.weight")
mid_fc3_weight_tensor7 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result8/cnnmodel.pkl","fc3.weight")
mid_fc3_weight_tensor8 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result9/cnnmodel.pkl","fc3.weight")
mid_fc3_weight_tensor9 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result1/cnnmodel.pkl","fc3.weight")
mid_fc3_weight_tensor10 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result10/cnnmodel.pkl","fc3.weight")
mid_fc3_weight_sum=tensoradd(mid_fc3_weight_tensor1,mid_fc3_weight_tensor2,mid_fc3_weight_tensor3,mid_fc3_weight_tensor4,mid_fc3_weight_tensor5,
          mid_fc3_weight_tensor6,mid_fc3_weight_tensor7,mid_fc3_weight_tensor8,mid_fc3_weight_tensor9,mid_fc3_weight_tensor10)

#---------------------------------min fc3 weight----------------------------------------------------
min_fc3_weight_tensor1 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result3/cnnmodel.pkl","fc3.weight")
min_fc3_weight_tensor2 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result3/cnnmodel.pkl","fc3.weight")
min_fc3_weight_tensor3 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result8/cnnmodel.pkl","fc3.weight")
min_fc3_weight_tensor4 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result9/cnnmodel.pkl","fc3.weight")
min_fc3_weight_tensor5 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result10/cnnmodel.pkl","fc3.weight")
min_fc3_weight_tensor6 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result1/cnnmodel.pkl","fc3.weight")
min_fc3_weight_tensor7 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result6/cnnmodel.pkl","fc3.weight")
min_fc3_weight_tensor8 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result6/cnnmodel.pkl","fc3.weight")
min_fc3_weight_tensor9 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-8/train_result2/cnnmodel.pkl","fc3.weight")
min_fc3_weight_tensor10 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result3/cnnmodel.pkl","fc3.weight")
min_fc3_weight_sum=tensoradd(min_fc3_weight_tensor1,min_fc3_weight_tensor2,min_fc3_weight_tensor3,min_fc3_weight_tensor4,min_fc3_weight_tensor5,
          min_fc3_weight_tensor6,min_fc3_weight_tensor7,min_fc3_weight_tensor8,min_fc3_weight_tensor9,min_fc3_weight_tensor10)



all_max_weight=tensoradd2(max_fc1_weight_sum,max_fc2_weight_sum,max_fc3_weight_sum)
all_mid_weight=tensoradd2(mid_fc1_weight_sum,mid_fc2_weight_sum,mid_fc3_weight_sum)
all_min_weight=tensoradd2(min_fc1_weight_sum,min_fc2_weight_sum,min_fc3_weight_sum)
lowestaccuracy1=tensoradd2(min_fc1_weight_tensor3.flatten(),min_fc2_weight_tensor3.flatten(),min_fc3_weight_tensor3.flatten())
lowestaccuracy2=tensoradd2(min_fc1_weight_tensor4.flatten(),min_fc2_weight_tensor4.flatten(),min_fc3_weight_tensor4.flatten())
lowestaccuracy3=tensoradd2(min_fc1_weight_tensor5.flatten(),min_fc2_weight_tensor5.flatten(),min_fc3_weight_tensor5.flatten())
lowestaccuracy4=tensoradd2(min_fc1_weight_tensor6.flatten(),min_fc2_weight_tensor6.flatten(),min_fc3_weight_tensor6.flatten())
lowestaccuracy5=tensoradd2(min_fc1_weight_tensor7.flatten(),min_fc2_weight_tensor7.flatten(),min_fc3_weight_tensor7.flatten())
lowestaccuracy6=tensoradd2(min_fc1_weight_tensor8.flatten(),min_fc2_weight_tensor8.flatten(),min_fc3_weight_tensor8.flatten())
lowestaccuracy7=tensoradd2(min_fc1_weight_tensor10.flatten(),min_fc2_weight_tensor10.flatten(),min_fc3_weight_tensor10.flatten())
#min_fc1_weight_tensor3,min_fc1_weight_tensor4,min_fc1_weight_tensor5,min_fc1_weight_tensor6,min_fc1_weight_tensor7,min_fc1_weight_tensor8,min_fc1_weight_tensor10
#print("mean:",format(np.mean(all_min_weight.tolist()),'.4f'))
print(mean_std(lowestaccuracy1))
print("mean1:",np.mean(lowestaccuracy1.tolist()),"std1:",np.std(lowestaccuracy1.tolist()))
print("mean2:",np.mean(lowestaccuracy2.tolist()),"std2:",np.std(lowestaccuracy2.tolist()))
print("mean3:",np.mean(lowestaccuracy3.tolist()),"std3:",np.std(lowestaccuracy3.tolist()))
print("mean4:",np.mean(lowestaccuracy4.tolist()),"std4:",np.std(lowestaccuracy4.tolist()))
print("mean5:",np.mean(lowestaccuracy5.tolist()),"std5:",np.std(lowestaccuracy5.tolist()))
print("mean6:",np.mean(lowestaccuracy6.tolist()),"std6:",np.std(lowestaccuracy6.tolist()))
print("mean7:",np.mean(lowestaccuracy7.tolist()),"std7:",np.std(lowestaccuracy7.tolist()))
csvfilesave('/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/meanstd/mean.csv',a)
#painthistoverlap(all_max_weight,all_mid_weight,all_min_weight,"All weights comparisons of all layers")
#plt.tight_layout()
#plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/All_layers_distribution_overlap.png")
#plt.show()

def addweight(path):
    f=open(path,'rb')
    dod=torch.load(f)
    torch.set_printoptions(threshold=np.inf) #Display all data for complex matrix 显示所有省略号的玩意
    tensor1=(dod["fc1.weight"]).flatten()
    tensor2=(dod["fc2.weight"]).flatten()
    tensor3=(dod["fc3.weight"]).flatten()
    concatenated_tensor = torch.cat((tensor1,tensor2,tensor3), dim=0).flatten()
    mean=np.mean(lowestaccuracy1.tolist())
    csvfilesave('/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/meanstd/mean.csv',mean)
    std=np.std(lowestaccuracy1.tolist())
    csvfilesave('/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/meanstd/std.csv',std)