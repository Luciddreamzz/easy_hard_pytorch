
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
#---------------------------------get weight----------------------------------------------------
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
    bins=np.arange(min(fc_weight).item(), max(fc_weight).item() +freedman_diaconis_rule(fc_weight), freedman_diaconis_rule(fc_weight))
    return bins
#---------------------------------add all tensor into one tensor----------------------------------------------------
def tensoradd(tensor1,tensor2,tensor3,tensor4,tensor5,tensor6,tensor7,tensor8,tensor9,tensor10):
# concatenate the two tensors along dimension 0
    concatenated_tensor = torch.cat((tensor1,tensor2,tensor3,tensor4,tensor5,tensor6,tensor7,tensor8,tensor9,tensor10), dim=0).flatten()
    return concatenated_tensor

#---------------------------------add all tensor into one tensor----------------------------------------------------
def tensoradd2(tensor1,tensor2,tensor3):
# concatenate the two tensors along dimension 0
    concatenated_tensor = torch.cat((tensor1,tensor2,tensor3), dim=0).flatten()
    return concatenated_tensor

def painthistcumulative(fc_weight_max, fc_weight_mid, fc_weight_min, figure_name):
    min_cumfreq = stats.cumfreq(fc_weight_min,numbins=100)
    x_min=min_cumfreq.lowerlimit+np.linspace(0,min_cumfreq.binsize*min_cumfreq.cumcount.size,min_cumfreq.cumcount.size)
    mid_cumfreq = stats.cumfreq(fc_weight_mid,numbins=100)
    x_mid=mid_cumfreq.lowerlimit+np.linspace(0,mid_cumfreq.binsize*mid_cumfreq.cumcount.size,mid_cumfreq.cumcount.size)
    max_cumfreq = stats.cumfreq(fc_weight_max,numbins=100)
    x_max=max_cumfreq.lowerlimit+np.linspace(0,max_cumfreq.binsize*max_cumfreq.cumcount.size,max_cumfreq.cumcount.size)
    fig, ax = plt.subplots()
    ax.bar(x_max, max_cumfreq.cumcount, width=0.02, color="gold", alpha=0.7, label='Max accuracy')
    ax.bar(x_mid, mid_cumfreq.cumcount, width=0.02, color="lightgreen", alpha=0.7, label='Mid accuracy')
    ax.bar(x_min, min_cumfreq.cumcount, width=0.02, color="lightskyblue", alpha=0.7, label='Min accuracy')
    ax.set_title('Weight Cumulative Histogram Overlap', fontsize=12)
    ax.set_xlabel('Weights', fontsize=10)
    ax.set_ylabel('Cumulative frequency', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/weight_cumulative_overlap1.png", dpi=300)
    plt.show()







#---------------------------------main function---------------------------------------------------- 
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
#lowestaccuracy1=tensoradd2(min_fc1_weight_tensor3,min_fc2_weight_tensor3,min_fc3_weight_tensor3)
#lowestaccuracy2=tensoradd2(min_fc1_weight_tensor4,min_fc2_weight_tensor4,min_fc3_weight_tensor4)
#lowestaccuracy3=tensoradd2(min_fc1_weight_tensor5,min_fc2_weight_tensor5,min_fc3_weight_tensor5)
#lowestaccuracy4=tensoradd2(min_fc1_weight_tensor6,min_fc2_weight_tensor6,min_fc3_weight_tensor6)
#lowestaccuracy5=tensoradd2(min_fc1_weight_tensor7,min_fc2_weight_tensor7,min_fc3_weight_tensor7)
#lowestaccuracy6=tensoradd2(min_fc1_weight_tensor8,min_fc2_weight_tensor8,min_fc3_weight_tensor8)
#lowestaccuracy7=tensoradd2(min_fc1_weight_tensor10,min_fc2_weight_tensor10,min_fc3_weight_tensor10)
#min_fc1_weight_tensor3,min_fc1_weight_tensor4,min_fc1_weight_tensor5,min_fc1_weight_tensor6,min_fc1_weight_tensor7,min_fc1_weight_tensor8,min_fc1_weight_tensor10
#print("mean:",format(np.mean(all_min_weight.tolist()),'.4f'))
#print("mean:",np.mean(all_min_weight.tolist()),np.std(all_min_weight.tolist()))
painthistcumulative(all_max_weight,all_mid_weight,all_min_weight,"All weights comparisons of all layers")
