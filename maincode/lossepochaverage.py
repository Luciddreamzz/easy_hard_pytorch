import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def load_weight1(path1, epoch_num):
    # Load the model that we saved at the end of the training loop 
    with open(path1, 'rb') as f1:
        dod1 = torch.load(f1)
    train_loss_epochs = [dod1['model_epoch{}'.format(i)]['train_loss'].item() for i in range(1,21)]
    test_acc = dod1['model_epoch20']['test_acc']
    return train_loss_epochs, test_acc

# Define the paths of the trained models
model_paths1 = ["/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result8/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result9/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result10/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result1/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result6/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result6/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result3/train_history_dic.pkl"]
print(type(model_paths1))
model_paths2 = ["/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result7/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result3/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result1/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result3/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result2/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result4/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-8/train_result2/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result8/train_history_dic.pkl"]
model_paths3 = ["/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result10/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result9/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result4/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result5/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result8/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result10/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result8/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result9/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result1/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result10/train_history_dic.pkl"]
model_paths4 = ["/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result5/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result6/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result5/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result5/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result6/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result8/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result10/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result2/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result2/train_history_dic.pkl",
"/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result10/train_history_dic.pkl"]
# Define the colors for the lines based on the accuracy bar
# 定义全局变量来存储所有模型的测试精度
def plot(model_path, ax, cmap, norm, cbar,name):      
    # Create an empty list to store the training loss of each model for each epoch
    train_loss = [[] for i in range(len(model_path))]
    test_acc = []

    # Load the training loss of each model for each epoch
    for i, path in enumerate(model_path):
        train_loss_epochs, acc = load_weight1(path, i)
        train_loss[i] = train_loss_epochs
        test_acc.append(acc)

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
cmap = plt.cm.PiYG
test_acc = []
for path in model_paths1 + model_paths2 + model_paths3:
    _, acc = load_weight1(path, 0)
    test_acc.append(acc)

ax.set_xlim(0, 19)
ax.set_ylim(0, 2.5)

ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

ax.set_xticks(list(range(0, 20, 1)) + [19])
norm = plt.Normalize(vmin=0, vmax=100)

# Add the colorbar
cbar=plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
cbar.set_label('Accuracy', fontsize=10)

# Plot the convergence curves for the 7 models with the lowest accuracy
plot(model_paths1, ax, cmap, norm, cbar, "min accuracy")
plot(model_paths2, ax, cmap, norm, cbar, "lowermid accuracy")
plot(model_paths3, ax, cmap, norm, cbar, "highermid accuracy")
plot(model_paths4, ax, cmap, norm, cbar, "max accuracy")
# plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result_process/plotvresult_b100l5r0001e20/convergence_average1.png", dpi=300)
plt.show()




