import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 获取模型路径
model_paths = []
for i in range(1, 101):
    for j in range(1, 11):
        path = "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-mnist/b100l5lr0001-{}/train_result{}/cnnmodel.pkl".format(i, j)
        model_paths.append(path)

model_paths1 = []
model_paths3 = []
model_paths4 = []

for path in model_paths:
    history_path = path.replace('cnnmodel.pkl', 'train_history_dic.pkl')
    with open(history_path, 'rb') as g:
        history = torch.load(g)
    test_acc = history['model_epoch20']['test_acc']
    if test_acc < 0.2:
        model_paths1.append(path)
    diff1 = abs(test_acc - 0.55)
    diff2 = abs(test_acc - 0.9)
    model_paths3.append((path, diff1))
    model_paths4.append((path, diff2))

model_paths3.sort(key=lambda x: x[1])
model_paths4.sort(key=lambda x: x[1])

model_paths3 = [x[0] for x in model_paths3[:100]]
model_paths4 = [x[0] for x in model_paths4[:100]]

# 定义函数计算两个模型之间的L2距离
def compute_l2_distance(tensor1, tensor2):
    return torch.norm(tensor1 - tensor2).item()

# 定义函数获取模型的fc1.weight
def get_fc1_weight(path):
    with open(path, 'rb') as f:
        model = torch.load(f)
        return model["fc1.weight"]

# 定义函数计算距离矩阵
def compute_distance_matrix(paths):
    num_models = len(paths)
    distance_matrix = np.zeros((num_models, num_models))

    for i in range(num_models):
        tensor_i = get_fc1_weight(paths[i])
        for j in range(num_models):
            tensor_j = get_fc1_weight(paths[j])
            distance_matrix[i, j] = compute_l2_distance(tensor_i, tensor_j)

    return distance_matrix

# 修改绘制热图的函数以适应距离矩阵
def plot_heatmap(ax, distance_matrix, title):
    sns.heatmap(distance_matrix, cmap='viridis', ax=ax)
    ax.set_title(title)

# 计算距离矩阵
distance_matrix1 = compute_distance_matrix(model_paths1)
distance_matrix3 = compute_distance_matrix(model_paths3)
distance_matrix4 = compute_distance_matrix(model_paths4)

# 创建一个 1x3 的子图布局
fig, axs = plt.subplots(1, 3, figsize=(30, 8))

# 绘制三个子图
plot_heatmap(axs[0], distance_matrix1, "FC1 Distance Matrix Heatmap for min acc")
plot_heatmap(axs[1], distance_matrix3, "FC1 Distance Matrix Heatmap for mid acc")
plot_heatmap(axs[2], distance_matrix4, "FC1 Distance Matrix Heatmap for max acc")

# 保存到本地
plt.tight_layout()
plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-mnist-plot/distance_heatmapsfc1.png")
plt.show()
