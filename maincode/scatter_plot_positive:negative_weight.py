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


def scatter_with_accuracy(x, y, accuracy, ax, vmin, vmax):
    im = ax.scatter(x, y, c=accuracy, cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax.set_xlabel('positive weight',fontsize=7)
    ax.set_ylabel('negative weight',fontsize=7)
    #ax.set_xlabel('mean',fontsize=7)
    #ax.set_ylabel('std',fontsize=7)
    ax.tick_params(axis='both', labelsize=6)
    #ax.set_title('Scatter Plot with Accuracy', fontsize=8)
    return im


def addweight(axs, path1, path2, vmin, vmax):
    f1 = open(path1,'rb')
    dod1 = torch.load(f1)
    f2 = open(path2,'rb')
    dod2 = torch.load(f2)
    torch.set_printoptions(threshold=np.inf) #Display all data for complex matrix
    
    tensor1 = (dod1["fc1.weight"]).flatten()
    accuracy = dod2['model_epoch20']['test_acc']
    fc1_positive = copy.deepcopy(tensor1)
    fc1_negative = copy.deepcopy(tensor1)
    fc1_positive_mean = fc1_positive[fc1_positive  >= 0].mean()
    fc1_positive_std = fc1_positive[fc1_positive  >= 0].std()
    
    fc1_negative_mean = fc1_negative[fc1_negative < 0].mean()
    fc1_negative_std = fc1_negative[fc1_negative < 0].std()
    
    tensor2 = (dod1["fc2.weight"]).flatten()
    fc2_positive = copy.deepcopy(tensor2)
    fc2_negative = copy.deepcopy(tensor2)
    fc2_positive_mean = fc2_positive[fc2_positive  >= 0].mean()
    fc2_positive_std = fc2_positive[fc2_positive  >= 0].std()
    
    fc2_negative_mean = fc2_negative[fc2_negative < 0].mean()
    fc2_negative_std = fc2_negative[fc2_negative < 0].std()

    tensor3 = (dod1["fc3.weight"]).flatten()
    fc3_positive = copy.deepcopy(tensor3)
    fc3_negative = copy.deepcopy(tensor3)
    fc3_positive_mean = fc3_positive[fc3_positive  >= 0].mean()
    fc3_positive_std = fc3_positive[fc3_positive  >= 0].std()
    fc3_negative_mean = fc3_negative[fc3_negative < 0].mean()
    fc3_negative_std = fc3_negative[fc3_negative < 0].std()
    
    all_positive_mean = torch.stack((fc1_positive_mean, fc2_positive_mean, fc3_positive_mean), dim=0).flatten().mean()
    all_negative_mean = torch.stack((fc1_negative_mean, fc2_negative_mean, fc3_negative_mean), dim=0).flatten().mean()
    all_positive_std = torch.stack((fc1_positive_std, fc2_positive_std, fc3_positive_std), dim=0).flatten().std()
    all_negative_std = torch.stack((fc1_negative_std, fc2_negative_std, fc3_negative_std), dim=0).flatten().std()
    
# plot positive weight&negative weight scatter
    #scatter_with_accuracy(fc1_positive_mean, fc1_negative_mean, accuracy, axs[0], vmin, vmax)
    #scatter_with_accuracy(fc2_positive_mean, fc2_negative_mean, accuracy, axs[1], vmin, vmax)
    #scatter_with_accuracy(fc3_positive_mean, fc3_negative_mean, accuracy, axs[2], vmin, vmax)
    #scatter_with_accuracy(all_positive_mean, all_negative_mean, accuracy, axs[3], vmin, vmax)

# plot positive weight&negative weight scatter
    scatter_with_accuracy(fc1_positive_std, fc1_negative_std, accuracy, axs[0], vmin, vmax)
    scatter_with_accuracy(fc2_positive_std, fc2_negative_std, accuracy, axs[1], vmin, vmax)
    scatter_with_accuracy(fc3_positive_std, fc3_negative_std, accuracy, axs[2], vmin, vmax)
    scatter_with_accuracy(all_positive_std, all_negative_std, accuracy, axs[3], vmin, vmax)

# plot positive weight mean-std scatter  
    #scatter_with_accuracy(fc1_positive_mean, fc1_positive_std, accuracy, axs[0], vmin, vmax)
    #scatter_with_accuracy(fc2_positive_mean, fc2_positive_std, accuracy, axs[1], vmin, vmax)
    #scatter_with_accuracy(fc3_positive_mean, fc3_positive_std, accuracy, axs[2], vmin, vmax)
    #scatter_with_accuracy(all_positive_mean, all_positive_std, accuracy, axs[3], vmin, vmax)
    
# plot negative weight mean-std scatter     
    #scatter_with_accuracy(fc1_negative_mean, fc1_negative_std, accuracy, axs[0], vmin, vmax)
    #scatter_with_accuracy(fc2_negative_mean, fc2_negative_std, accuracy, axs[1], vmin, vmax)
    #scatter_with_accuracy(fc3_negative_mean, fc3_negative_std, accuracy, axs[2], vmin, vmax)
    #scatter_with_accuracy(all_negative_mean, all_negative_std, accuracy, axs[3], vmin, vmax)
    

fig, axs = plt.subplots(nrows=1, ncols=4, dpi=300,figsize=(15, 3))

# define the minimum and maximum values for the colorbar
vmin = 0.0
vmax = 1.0

# loop through each set of files to plot
for i in range(10):
    for j in range(10):
        path1 = f"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-{i+1}/train_result{j+1}/cnnmodel.pkl"
        path2 = f"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-{i+1}/train_result{j+1}/train_history_dic.pkl"
        addweight(axs, path1, path2, vmin, vmax)

# set the colorbar
cbar = fig.colorbar(axs[0].collections[0], ax=axs, orientation='vertical', shrink=0.75)
cbar.ax.set_ylabel('Accuracy')
plt.tight_layout()
plt.subplots_adjust(right=0.75)
#plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result_process/plotvresult_b100l5r0001e20/mean_scatter_positive_negative.png", dpi=300)
plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result_process/plotvresult_b100l5r0001e20/std_scatter_positive_negative.png", dpi=300)
#plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result_process/plotvresult_b100l5r0001e20/mean_std_scatter_positive.png", dpi=300)
#plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result_process/plotvresult_b100l5r0001e20/mean_std_scatter_negative.png", dpi=300)
plt.show()
