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


def csvfilesave(csvfile_path,list_name):
    csvFile = open(csvfile_path, "a+",)
    try:
        writer = csv.writer(csvFile)
        writer.writerow(list_name)
    finally:
        csvFile.close() 

def addgradient(path):
    f=open(path,'rb')
    dod=torch.load(f)
    torch.set_printoptions(threshold=np.inf) #Display all data for complex matrix 显示所有省略号的玩意
    tensor1=(dod["model_epoch20 fc1.weight"])
    tensor2=(dod["model_epoch20 fc2.weight"])
    tensor3=(dod["model_epoch20 fc3.weight"])
    start_index1 = tensor1.find("[[")
    end_index1 = tensor1.rfind("]]") + 2
    desired_string1 = tensor1[start_index1:end_index1]
    # define the string representation of the array
    # create a numpy array from the string representation
    array1 = np.array(eval(desired_string1))
# convert the numpy array to a tensor
    gradientfc1 = (torch.from_numpy(array1)).flatten()
    
    
    start_index2 = tensor2.find("[[")
    end_index2 = tensor2.rfind("]]") + 2
    desired_string2 = tensor2[start_index2:end_index2]
    # define the string representation of the array
    # create a numpy array from the string representation
    array2 = np.array(eval(desired_string2))
# convert the numpy array to a tensor
    gradientfc2 = (torch.from_numpy(array2)).flatten()
    
    
    start_index3 = tensor3.find("[[")
    end_index3 = tensor3 .rfind("]]") + 2
    desired_string3 = tensor3[start_index3:end_index3]
    # define the string representation of the array
    # create a numpy array from the string representation
    array3  = np.array(eval(desired_string3))
# convert the numpy array to a tensor
    gradientfc3 = (torch.from_numpy(array3)).flatten()
    
    
    concatenated_tensor = torch.cat((gradientfc1,gradientfc2,gradientfc3), dim=0).flatten()
    mean=[np.mean(gradientfc3.tolist())]
    csvfilesave('/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result_process/meanstdgradientcsvresult_b100l5r0001e20/meanfc3.csv',mean)
    std=[np.std(gradientfc3.tolist())]
    csvfilesave('/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result_process/meanstdgradientcsvresult_b100l5r0001e20/stdfc3.csv',std)
    
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result1/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result2/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result3/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result4/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result5/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result6/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result7/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result8/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result9/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result10/gradient_dic.pkl")

addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result1/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result2/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result3/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result4/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result5/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result6/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result7/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result8/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result9/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result10/gradient_dic.pkl")

addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result1/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result2/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result3/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result4/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result5/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result6/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result7/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result8/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result9/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result10/gradient_dic.pkl")

addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result1/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result2/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result3/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result4/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result5/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result6/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result7/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result8/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result9/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result10/gradient_dic.pkl")

addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result1/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result2/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result3/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result4/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result5/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result6/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result7/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result8/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result9/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result10/gradient_dic.pkl")

addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result1/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result2/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result3/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result4/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result5/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result6/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result7/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result8/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result9/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result10/gradient_dic.pkl")

addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result1/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result2/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result3/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result4/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result5/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result6/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result7/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result8/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result9/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result10/gradient_dic.pkl")

addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-8/train_result1/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-8/train_result2/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-8/train_result3/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-8/train_result4/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-8/train_result5/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-8/train_result6/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-8/train_result7/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-8/train_result8/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-8/train_result9/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-8/train_result10/gradient_dic.pkl")

addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result1/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result2/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result3/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result4/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result5/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result6/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result7/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result8/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result9/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result10/gradient_dic.pkl")

addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result1/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result2/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result3/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result4/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result5/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result6/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result7/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result8/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result9/gradient_dic.pkl")
addgradient("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result10/gradient_dic.pkl")

