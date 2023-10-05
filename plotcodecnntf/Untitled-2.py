import re
import torch
import numpy as np 
import matplotlib.mlab as mlab 
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
from matplotlib.collections import LineCollection
import seaborn as sns 
import matplotlib as mpl 
from torch import tensor
from matplotlib.collections import PolyCollection
torch.set_printoptions(threshold=np.inf)

def nodesproperty(dictionary, layer1, layer2, flag):
    z_sum = 0
    z1 = []
    if flag == 1:
        s = dictionary.sum(axis=1)  
        for i in range(layer2):
            for j in range(layer1):
                z = (abs(dictionary[i][j]) / abs(s[i]))**2
                z_sum += z
            z1.append('{:.4f}'.format(z_sum))
            z_sum = 0
    else:
        s = dictionary.sum(axis=0)
        for i in range(layer2):
            for j in range(layer1):
                z = (abs(dictionary[j][i]) / abs(s[i]))**2
                z_sum += z
            z1.append('{:.4f}'.format(z_sum))
            z_sum = 0
    return sum([float(x) if x != 'nan' else 0 for x in z1]), torch.sum(s).item()

input_size = 28 * 28
output_size = 10
node1 = 5
node2 = 5

def extract_weights(file_path, layer_name):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    capture = False
    epoch = None
    weight_lines = ''

    for line in lines:
        if "epoch" in line:
            epoch = int(re.search(r'\d+', line).group())
            continue

        if layer_name in line:
            capture = True
            weight_lines = ''
            continue

        if capture:
            weight_lines += line.strip()
            if weight_lines.count('[') == weight_lines.count(']'):
                weight_lines = weight_lines[7:-1]
                weights = eval(weight_lines)
                weight_tensor = torch.tensor(weights)
                capture = False
                yield epoch, weight_tensor


cmap = plt.cm.jet
fig, ax = plt.subplots(dpi=300, figsize=(10, 6))
# ax.set_ylim([0, 100]) 

model_paths = []    
for i in range(1, 101):
    for j in range(1, 11):
        path = "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-mnist/b100l5lr0001-{}/train_result{}/weights.txt".format(i, j)
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
    history_path = path.replace('weights.txt', 'train_history_dic.pkl')
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
model_paths = [model_paths1, model_paths2, model_paths3, model_paths4]



# Group all model paths
all_model_paths = [model_paths1,model_paths2,model_paths3,model_paths4]

for model_paths in all_model_paths:
    all_node_disparity_values = []

    # Track all the epoch values (assuming they are the same for all paths)
    epoch_values = []
    final_epoch_accuracy_values = []

    for i, path in enumerate(model_paths):  
        weight_generator = extract_weights(path, "fc1.weight")
        experiment_node_disparity_values = []
        for k in range(1, 21):
            epoch, tensor1 = next(weight_generator)
            if i == 0:  # Only on the first iteration, we save the epoch values
                epoch_values.append(epoch)
            z_values, _ = nodesproperty(tensor1, input_size, node1, 1)
            if isinstance(z_values, float):
                z_values = [z_values]
            # Convert to numpy array and filter out inf values
            z_values = np.array(z_values)
            z_values = z_values[np.isfinite(z_values)]
            
            # Apply clipping
            z_values = np.clip(z_values, None, 1000000)  # Clip at upper bound 1000000

            # Apply log transformation
            z_values = np.log1p(z_values)

            experiment_node_disparity_values.extend(z_values.tolist())
            print(experiment_node_disparity_values)
        # Only add non-empty disparity values
        if experiment_node_disparity_values:
            all_node_disparity_values.append(experiment_node_disparity_values)

        # Get final epoch test accuracy
        history_path = path.replace('weights.txt', 'train_history_dic.pkl')
        with open(history_path, 'rb') as g:
            history = torch.load(g)
        final_epoch_accuracy_values.append(history['model_epoch20']['test_acc']) # Assuming 'model_epoch20' is always present

    # Convert to numpy array for easier manipulation
    all_node_disparity_values = np.array(all_node_disparity_values)

    # Calculate the minimum, maximum and mean node disparity values for each epoch
    min_values = np.min(all_node_disparity_values, axis=0)
    max_values = np.max(all_node_disparity_values, axis=0)
    mean_values = np.mean(all_node_disparity_values, axis=0)

    # Determine color from average final epoch accuracy
    c = cmap(np.mean(final_epoch_accuracy_values))

    # Create shaded range graph
    ax.fill_between(epoch_values, min_values, max_values, color=c, alpha=0.3)

    # Plot the mean line
    ax.plot(epoch_values, mean_values, color=c, alpha=0.7)

# Other plot settings
ax.set_xlabel('Epoch', fontsize=10)
ax.set_ylabel('Node Disparity', fontsize=10) # Change ylabel here
ax.tick_params(axis='both', labelsize=8)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
cbar = plt.colorbar(sm)

plt.tight_layout()
plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-mnist-plot/nodedisparity_lineplot_fc11111.png", dpi=300)
plt.show()






