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

# 定义函数计算两个模型之间的余弦相似度
def compute_cosine_similarity(tensor1, tensor2):
    tensor1 = tensor1.flatten()
    tensor2 = tensor2.flatten()
    cosine_similarity = torch.nn.functional.cosine_similarity(tensor1, tensor2, dim=0)
    return cosine_similarity.item()

# 定义函数获取模型的fc1.weight
def get_fc1_weight(path):
    with open(path, 'rb') as f:
        model = torch.load(f)
        return model["fc2.weight"]

# 定义函数计算相似性矩阵
def compute_similarity_matrix(paths):
    num_models = len(paths)
    similarity_matrix = np.zeros((num_models, num_models))

    for i in range(num_models):
        tensor_i = get_fc1_weight(paths[i])
        for j in range(num_models):
            tensor_j = get_fc1_weight(paths[j])
            similarity_matrix[i, j] = compute_cosine_similarity(tensor_i, tensor_j)

    return similarity_matrix

def plot_heatmap(ax, similarity_matrix, title):
    sns.heatmap(similarity_matrix, cmap='viridis', vmin=-1, vmax=1, ax=ax)
    ax.set_title(title)

# 计算相似性矩阵
similarity_matrix1 = compute_similarity_matrix(model_paths1)
similarity_matrix3 = compute_similarity_matrix(model_paths3)
similarity_matrix4 = compute_similarity_matrix(model_paths4)

# 创建一个 1x3 的子图布局
fig, axs = plt.subplots(1, 3, figsize=(30, 8))

# 绘制三个子图
plot_heatmap(axs[0], similarity_matrix1, "FC2 Similarity Matrix Heatmap for min acc")
plot_heatmap(axs[1], similarity_matrix3, "FC2 Similarity Matrix Heatmap for mid acc")
plot_heatmap(axs[2], similarity_matrix4, "FC2 Similarity Matrix Heatmap for max acc")

# 保存到本地
plt.tight_layout()
plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/resultb100l5lr0001-mnist-plot/similarity_heatmapsFC2.png")
plt.show()
