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
    im = ax.scatter(x, y, c=accuracy, cmap='RdPu', vmin=vmin, vmax=vmax)
    ax.set_xlabel('positive node dispaity',fontsize=7)
    ax.set_ylabel('negative node dispaity',fontsize=7)
    #ax.set_xlabel('mean',fontsize=7)
    #ax.set_ylabel('std',fontsize=7)
    axs[3].set_xlim(400, 600)
    axs[3].set_ylim(300, 700)
    ax.tick_params(axis='both', labelsize=6)
    #ax.set_title('Scatter Plot with Accuracy', fontsize=8)
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

def addweight(axs, path1, path2, vmin, vmax):
    f = open(path1,'rb')
    dod1 = torch.load(f)
    f = open(path2,'rb')
    dod2 = torch.load(f)
    torch.set_printoptions(threshold=np.inf) #Display all data for complex matrix
    
    tensor1 = (dod1["fc1.weight"])
    tensor2 = (dod1["fc2.weight"])
    tensor3 = (dod1["fc3.weight"])
    accuracy = dod2['model_epoch20']['test_acc']
    
    #z1:Disparity of the neural nodes-- input edges
    #z2:hidden layer1-hidden layer2
    #z3:hidden layer2-output layer
    #s1:Strength of the neural nodes-- input edges
    #s2:hidden layer1-hidden layer2
    #s3:hidden layer2-output layer
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
    zsum1=[z4,z5,z6]
    #Disparity and strength of the neural nodes-- input edges --negative
    z7,s7=nodesproperty(fc1_negative,input_size,node1,1)
    z8,s8=nodesproperty(fc2_negative,node1,node2,1)
    z9,s9=nodesproperty(fc3_negative,node2,output_size,1)  
    zsum2=[z7,z8,z9]
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
    #scatter_with_accuracy(z4, z7, accuracy, axs[0], vmin, vmax)
    #scatter_with_accuracy(z5, z8, accuracy, axs[1], vmin, vmax)
    #scatter_with_accuracy(z6, z9, accuracy, axs[2], vmin, vmax)
    #scatter_with_accuracy(z4, z7, accuracy, axs[3], vmin, vmax)
    #scatter_with_accuracy(z5, z8, accuracy, axs[3], vmin, vmax)
    #scatter_with_accuracy(z6, z9, accuracy, axs[3], vmin, vmax)

# plot sactter of output edges sum node disparity and sum node strength--positive part
    scatter_with_accuracy(o4, o7, accuracy, axs[0], vmin, vmax)
    scatter_with_accuracy(o5, o8, accuracy, axs[1], vmin, vmax)
    scatter_with_accuracy(o6, o9, accuracy, axs[2], vmin, vmax)
    scatter_with_accuracy(o4, o7, accuracy, axs[3], vmin, vmax)
    scatter_with_accuracy(o5, o8, accuracy, axs[3], vmin, vmax)
    scatter_with_accuracy(o6, o9, accuracy, axs[3], vmin, vmax)

fig, axs = plt.subplots(nrows=1, ncols=4, dpi=300,figsize=(15, 3))

# define the minimum and maximum values for the colorbar
vmin = 0.0
vmax = 1.0
#hyperparameters of networks
input_size= 28*28
output_size=10
node1=5
node2=5
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
plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result_process/plotvresult_b100l5r0001e20/sactter_node_positive and negative disparity_output_edges.png", dpi=300)
#plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result_process/plotvresult_b100l5r0001e20/sactter_node_disparity_and_strength_input_edges_positive.png", dpi=300)
#plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result_process/plotvresult_b100l5r0001e20/sactter_node_disparity_and_strength_input_edges_negative.png", dpi=300)
#plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result_process/plotvresult_b100l5r0001e20/sactter_node_disparity_and_strength_output_edges.png", dpi=300)
#plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result_process/plotvresult_b100l5r0001e20/sactter_node_disparity_and_strength_output_edges_positive.png", dpi=300)
#plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result_process/plotvresult_b100l5r0001e20/sactter_node_disparity_and_strength_output_edges_negative.png", dpi=300)
plt.show()

