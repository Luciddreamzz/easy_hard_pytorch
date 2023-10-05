import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def load_weight1(path):
    # Load the model that we saved at the end of the training loop 
    with open(path, 'rb') as f1:
        dod1 = torch.load(f1)
    train_loss_epochs = [dod1['model_epoch{}'.format(i)]['train_loss'].item() for i in range(1,21)]
    test_acc = dod1['model_epoch20']['test_acc']
    return train_loss_epochs, test_acc

model_paths = []
for i in range(1, 101):
    for j in range(1, 11):
        path = "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-cifar10/b100l5lr0001-{}/train_result{}/train_history_dic.pkl".format(i, j)
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
    with open(path, 'rb') as f:
        # Load cnnmodel and history dictionary from the file
        cnnmodel = torch.load(f)
        history_path = path.replace('cnnmodel.pkl', 'train_history_dic.pkl')
        with open(history_path, 'rb') as g:
            history = torch.load(g)
        # Get test accuracy from the history dictionary
        test_acc = history['model_epoch20']['test_acc']
        # compute the difference between test_acc and 0.65
        diff1 = abs(test_acc - 0)
        # append the path and the difference to the corresponding lists
        model_paths3.append(path)
        test_acc_diff1.append(diff1)
        
        diff2 = abs(test_acc - 1)
        # append the path and the difference to the corresponding lists
        model_paths4.append(path)
        test_acc_diff2.append(diff2)
# get the indices of the 100 smallest differences
indices1 = sorted(range(len(test_acc_diff1)), key=lambda i: test_acc_diff1[i])[:20]
# use the indices to get the corresponding paths
model_paths3 = [model_paths3[i] for i in indices1]
indices2 = sorted(range(len(test_acc_diff2)), key=lambda i: test_acc_diff2[i])[:20]
# use the indices to get the corresponding paths
model_paths4 = [model_paths4[i] for i in indices2]
print(len(model_paths3),len(model_paths4))

# Define the colors for the lines based on the accuracy bar
def plot(model_path, ax, cmap, norm, cbar,name):      
    # Create an empty list to store the training loss of each model for each epoch
    train_loss = [[] for i in range(len(model_path))]
    test_acc = []

    # Load the training loss of each model for each epoch
    for i, path in enumerate(model_path):
        train_loss_epochs, acc = load_weight1(path)
        train_loss[i] = train_loss_epochs
        test_acc.append(acc)
        print(acc)
    # Calculate the mean and standard deviation of the training loss at each epoch across all models
    mean_train_loss = np.mean(train_loss, axis=0)
    std_train_loss = np.std(train_loss, axis=0)
    # Calculate the mean test accuracy across all models
    mean_test_acc = np.mean(test_acc)
    # Plot the average convergence curve as a solid line and the standard deviation as a shaded region around the mean line
    ax.plot(mean_train_loss, label=name, color=cmap(norm(mean_test_acc*100)), linewidth=2)
    ax.fill_between(range(len(mean_train_loss)), mean_train_loss-std_train_loss, mean_train_loss+std_train_loss, alpha=0.2, color=cmap(norm(mean_test_acc*100)))


    # Plot the convergence curves for all models as dashed lines
    for i, loss in enumerate(train_loss):
        ax.plot(loss, linestyle='dashed', color=cmap(norm(mean_test_acc*100)), alpha=0.5)

    # Set the title and labels for the graph
    ax.set_title('', fontsize=14)
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)

    # Add the legend for the graph
    ax.legend(fontsize=10)

    # Add the scatter plot for the colorbar
    sc = ax.scatter([], [], c=[], cmap=cmap, norm=norm)
    cbar.update_normal(sc)

    return sc

# Create a single figure to plot all curves
fig, ax = plt.subplots(figsize=(10, 8))

# Define the colors for the lines based on the accuracy bar
cmap = plt.cm.jet
test_acc = []
for path in model_paths3 + model_paths4:
    _, acc = load_weight1(path)
    test_acc.append(acc)

ax.set_xlim(0, 19)
ax.set_ylim(1.5, 2.5)

# ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

ax.set_xticks(list(range(0, 20, 1)) + [19])
norm = plt.Normalize(vmin=20, vmax=40)

# Add the colorbar
cbar=plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
cbar.set_label('Accuracy', fontsize=10)

# Plot the convergence curves for the 7 models with the lowest accuracy
plot(model_paths3, ax, cmap, norm, cbar, "min accuracy")
# plot(model_paths2, ax, cmap, norm, cbar, "mid accuracy")
plot(model_paths4, ax, cmap, norm, cbar, "max accuracy")
plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-cifar10plot/convergence_average.png", dpi=300)
plt.show()


 

