#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import transforms, datasets




#def draw_pic(index, cost_data):
#    plt.ion()
#    plt.plot(index, cost_data,color='dodgerblue', marker='')
#    plt.title("cost figure")
#    plt.xlabel('item_time')
#    plt.ylabel('cost')
#    plt.show() 
#    plt.pause(0.000001)

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

def load_data(batch_size, valid_size=0.2, num_workers=1):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = datasets.ImageFolder(root='/Users/jianqiaolong/Downloads/hr-mnist-pytorch/data/tiny-imagenet-200', transform=transform)
    test_dataset = datasets.ImageFolder(root='/Users/jianqiaolong/Downloads/hr-mnist-pytorch/data/tiny-imagenet-200', transform=transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_index, valid_index = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, drop_last=True)
    valid_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, test_loader, valid_loader


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, node1)
        self.fc2 = nn.Linear(node1, node2)
        self.fc3 = nn.Linear(node2, output_size)

    def forward(self, x):
        x = x.view(-1,input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)

    @staticmethod
    def check_opt(opt, lr):
        if opt == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=lr)
        elif opt == "Adam":
            optimizer = optim.Adam(net.parameters(), lr=lr, eps=1e-08, betas=(0.9, 0.999))
        elif opt == 'Adagrad':
            optimizer = optim.Adagrad(net.parameters(), lr=lr, lr_decay=0.5)
        else:
            raise ValueError('You Enter A Wrong Optimizer!')
        return optimizer

    def train(self, epoch_num, lr, criterion=nn.CrossEntropyLoss(), opt='SGD'):
        start_time = time.time()
        optimizer = self.check_opt(opt, lr)
        for i in range(epoch_num):
            for data, label in train_loader:
                data = torch.autograd.Variable(data)
                label = torch.autograd.Variable(label)
                out = net(data)
                optimizer.zero_grad()
                loss = criterion(out, label)
                loss.backward()
                optimizer.step()
            
            for data,label in valid_loader:
                data = torch.autograd.Variable(data)
                label = torch.autograd.Variable(label)
                out = net(data)
                optimizer.zero_grad()
                valid_loss = criterion(out, label)
                valid_loss.backward()
                optimizer.step()
                
            for data,label in test_loader:
                data = torch.autograd.Variable(data)
                label = torch.autograd.Variable(label)
                out = net(data)
                optimizer.zero_grad()
                test_loss = criterion(out, label)
                test_loss.backward()
                optimizer.step()

        train_time = time.time() - start_time
        return train_time


    def test(self, data_set):
        correct = 0
        total = 0
        for batch_i, data in enumerate(data_set):
            inputs, labels = data
            outputs = net(inputs)
            _,predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        accuracy = correct / total
        return accuracy

    def run_demo(self,input_data):
        output_data = net(input_data)
        result = torch.max(output_data, 1)
        return output_data, result



if __name__ == '__main__':
    batch_size = 100
    epoch = 20         
    opt = 'Adam'
    learning_rate = 0.001
    valid_size = 0.1 
    num_workers=0
    input_size= 3*64*64
    output_size=200
    node1=5
    node2=5
    train_loader, test_loader, valid_loader = load_data(batch_size)
    criterion = nn.CrossEntropyLoss()
    net =Mlp()
    train_time = net.train(epoch_num=epoch, lr=learning_rate, opt=opt, criterion=criterion)
    torch.save(net, '/Users/jianqiaolong/Downloads/hr-mnist-pytorch/cnnmodel.pkl')  # 保存整个模型
    print("The train model is saved")
    print("The paramenters of the model have been saved")

    train_accuracy = net.test(train_loader)
    test_accuracy = net.test(test_loader)
    valid_accuracy= net.test(valid_loader)
    print("Training the model...")
    
    detail = """
**************************************************************
            The training has been completed!
--------------------------------------------------------------   
            detail:
                epoch:{}       learning rate:{}
                batch:{}       optimizer:{}
--------------------------------------------------------------
            net struct:
{}
--------------------------------------------------------------
            train_accurancy:{}%
            test_accurancy:{}%  
            valid_accurancy:{}% 
            train_time:{}s
**************************************************************  
    """.format(epoch, learning_rate, batch_size, opt, net, train_accuracy*100, test_accuracy*100, valid_accuracy*100, train_time)
    print(detail)
