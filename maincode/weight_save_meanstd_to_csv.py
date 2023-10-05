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

def addweight(path):
    f=open(path,'rb')
    dod=torch.load(f)
    torch.set_printoptions(threshold=np.inf) #Display all data for complex matrix 显示所有省略号的玩意
    tensor1=(dod["fc1.weight"]).flatten()
    tensor2=(dod["fc2.weight"]).flatten()
    tensor3=(dod["fc3.weight"]).flatten()
    concatenated_tensor = torch.cat((tensor1,tensor2,tensor3), dim=0).flatten()
    mean=[np.mean(tensor3.tolist())]
    csvfilesave('/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/meanstd/meanfc3.csv',mean)
    std=[np.std(tensor3.tolist())]
    csvfilesave('/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/meanstd/stdfc3.csv',std)
    
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result1/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result2/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result3/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result4/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result5/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result6/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result7/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result8/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result9/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result10/cnnmodel.pkl")

addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result1/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result2/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result3/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result4/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result5/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result6/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result7/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result8/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result9/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result10/cnnmodel.pkl")

addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result1/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result2/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result3/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result4/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result5/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result6/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result7/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result8/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result9/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result10/cnnmodel.pkl")

addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result1/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result2/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result3/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result4/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result5/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result6/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result7/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result8/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result9/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result10/cnnmodel.pkl")

addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result1/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result2/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result3/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result4/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result5/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result6/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result7/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result8/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result9/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result10/cnnmodel.pkl")

addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result1/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result2/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result3/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result4/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result5/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result6/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result7/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result8/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result9/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result10/cnnmodel.pkl")

addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result1/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result2/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result3/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result4/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result5/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result6/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result7/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result8/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result9/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result10/cnnmodel.pkl")

addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-8/train_result1/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-8/train_result2/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-8/train_result3/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-8/train_result4/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-8/train_result5/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-8/train_result6/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-8/train_result7/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-8/train_result8/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-8/train_result9/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-8/train_result10/cnnmodel.pkl")

addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result1/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result2/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result3/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result4/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result5/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result6/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result7/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result8/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result9/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result10/cnnmodel.pkl")

addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result1/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result2/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result3/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result4/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result5/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result6/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result7/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result8/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result9/cnnmodel.pkl")
addweight("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result10/cnnmodel.pkl")