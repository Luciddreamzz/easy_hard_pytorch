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




  
    
#---------------------------------Filter tensors based on mask----------------------------------------------------
#Filter tensors based on mask. 根据mask对张量进行筛选
#z is the matrix want to be filter
def filter_mask(a):
    # get the mask
    less_zero= (a < 0)
    larger_zero = (a > 0)
    # output: select by mask
    return len(torch.masked_select(a, less_zero)),len(torch.masked_select(a, larger_zero)),
#---------------------------------载入已有权重模型----------------------------------------------------
def load_weight(path1,path2): 
    # Load the model that we saved at the end of the training loop 
    f1 = open(path1,'rb')
    dod1 = torch.load(f1)
    f2 = open(path2,'rb')
    dod2 = torch.load(f2)
    accuracy = dod2['model_epoch20']['test_acc']
    torch.set_printoptions(threshold=np.inf) #Display all data for complex matrix
    fc1_weight=dod1["fc1.weight"]
    fc2_weight=dod1["fc2.weight"]
    fc3_weight=dod1["fc3.weight"]
    # output: select by mask
    print("fc1_weight<0,fc1_weight>0:",filter_mask(fc1_weight)) 
    print("fc2_weight<0,fc2_weight>0:",filter_mask(fc2_weight)) 
    print("fc3_weight<0,fc3_weight>0:",filter_mask(fc3_weight))
    print("total_weight<0,total_weight>0:",filter_mask(fc1_weight)[0]+filter_mask(fc2_weight)[0]+filter_mask(fc3_weight)[0],filter_mask(fc1_weight)[1]+filter_mask(fc2_weight)[1]+filter_mask(fc3_weight)[1])
def load_weight1(path1): 
    # Load the model that we saved at the end of the training loop 
    f1 = open(path1,'rb')
    dod1 = torch.load(f1)
    print(dod1['model_epoch1']['test_loss'].item())
#---------------------------------main function---------------------------------------------------- 
if __name__ == '__main__':
    print("when accuracy=88.65%:")
    load_weight1("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001/train_result1/train_history_dic.pkl")
