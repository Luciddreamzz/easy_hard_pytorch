import pandas as pd
import matplotlib.pyplot as plt

def paintplot(figure, path, mean, std, figurename, n):
    ax = figure.add_subplot(1, 4, n)
    file = pd.read_excel(path)
    df = pd.DataFrame(file)
    accuracy_normalized = (df['accuracy'] - df['accuracy'].min()) / (df['accuracy'].max() - df['accuracy'].min())
    cmap = plt.cm.get_cmap('jet')
    scatter = ax.scatter(df[mean], df[std], c=accuracy_normalized, cmap=cmap)
    ax.tick_params(axis='both', labelsize=6)
    ax.set_title(figurename, fontsize=8)
    ax.set_xlabel("mean", fontsize=5)
    ax.set_ylabel("std", fontsize=5)
    # set the x and y limits for the current subplot
    if n == 1:
        plt.xlim(xmin=-0.01, xmax=0.011)
        plt.ylim(ymin=-0.001, ymax=0.032)

    return scatter

figure = plt.figure(figsize=(12, 3), dpi=300)

scatter1 = paintplot(figure, '/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result_process/meanstdgradientxlsvresult_b100l5r0001e20/meanstdfc1.xlsx', 'fc1mean', 'fc1std', "mean-std-accuracy fc1 scatter plot", 1)
scatter2 = paintplot(figure, '/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result_process/meanstdgradientxlsvresult_b100l5r0001e20/meanstdfc2.xlsx', 'fc2mean', 'fc2std', "mean-std-accuracy fc2 scatter plot", 2)
scatter3 = paintplot(figure, '/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result_process/meanstdgradientxlsvresult_b100l5r0001e20/meanstdfc3.xlsx', 'fc3mean', 'fc3std', "mean-std-accuracy fc3 scatter plot", 3)
scatter4 = paintplot(figure, '/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result_process/meanstdgradientxlsvresult_b100l5r0001e20/meanstdall.xlsx', 'mean', 'std', "mean-std-accuracy all scatter plot", 4)

plt.tight_layout()

# Create a colorbar for the accuracy rate
cax = figure.add_axes([0.93, 0.2, 0.01, 0.6])
cb = plt.colorbar(scatter1, cax=cax)
cb.set_label('Accuracy', fontsize=8)
cb.ax.tick_params(labelsize=6)

plt.subplots_adjust(right=0.9) # add padding to the right side of the figure

plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result_process/plotvresult_b100l5r0001e20/mean_std_scatter_gradient_all.png", dpi=300)
plt.show()