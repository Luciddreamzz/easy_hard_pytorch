import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
def load_weight1(path1, epoch_num):
    # Load the model that we saved at the end of the training loop 
    with open(path1, 'rb') as f1:
        dod1 = torch.load(f1)
    train_loss_epochs = [dod1['model_epoch{}'.format(i)]['train_loss'].item() for i in range(1,21)]
    test_acc = dod1['model_epoch20']['test_acc']
    return train_loss_epochs, test_acc

# Define the paths of the trained models
model_paths = []
for i in range(1, 101):
    for j in range(1, 11):
        path = "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-fashionmnist/b100l5lr0001-{}/train_result{}/train_history_dic.pkl".format(i, j)
        model_paths.append(path)
      
# Create an empty list to store the training loss of each model for each epoch
train_loss = [[] for i in range(len(model_paths))]
test_acc = []

# Load the training loss of each model for each epoch
for i, path in enumerate(model_paths):
    train_loss_epochs, acc = load_weight1(path, i)
    train_loss[i] = train_loss_epochs
    test_acc.append(acc)
plt.autoscale()
# Define the colors for the lines based on the accuracy bar
norm = mcolors.Normalize(vmin=0, vmax=100)
cmap = plt.cm.jet
sm = ScalarMappable(norm=norm, cmap=cmap)
    
# Plot the convergence curves for all models on the same graph
fig, ax = plt.subplots(figsize=(10, 8))
for i, loss in enumerate(train_loss):
    color = sm.to_rgba(test_acc[i] * 100)  # 将测试准确率转换为0-100范围内的百分比
    ax.plot(loss, label='model {}'.format(i+1), color=color)

ax.set_xlim(0, 19)
ax.set_ylim(0, 2.5)


ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

ax.set_xticks(list(range(0, 20, 1)) + [19])
ax.set_xlabel('Epoch', fontsize=10)
ax.set_ylabel('Loss', fontsize=10)

# Add the legend for the graph
#ax.legend(fontsize=3)

# Add the colorbar
cbar = plt.colorbar(sm)
cbar.set_label('Accuracy', fontsize=10)
plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-fashionmnist-plot/convergence.png", dpi=300)
plt.show()

#——————————————average---------------
""" # Calculate the mean and standard deviation of the training loss at each epoch across all models
mean_train_loss = np.mean(train_loss, axis=0)
std_train_loss = np.std(train_loss, axis=0)

# Plot the average convergence curve as a solid line and the standard deviation as a shaded region around the mean line
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(mean_train_loss, label='Average', color='black', linewidth=2)
ax.fill_between(range(len(mean_train_loss)), mean_train_loss-std_train_loss, mean_train_loss+std_train_loss, alpha=0.2, color='black')

# Plot the convergence curves for all models as dashed lines
for i, loss in enumerate(train_loss):
    ax.plot(loss, linestyle='dashed', color=colors[i], alpha=0.5)

# Set the title and labels for the graph
ax.set_title('Convergence Curves for 100 Trained Models', fontsize=14)
ax.set_xlabel('Epoch', fontsize=10)
ax.set_ylabel('Train Loss', fontsize=10)

# Add the legend for the graph
ax.legend(fontsize=10)

# Add the colorbar
sc = plt.scatter([],[], c=[], cmap=cmap, norm=norm)
cbar=plt.colorbar(sc)
cbar.set_label('Accuracy', fontsize=10)

plt.show() """
