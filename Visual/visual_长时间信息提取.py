import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import numpy as np
import warnings
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Times New Roman'  # 设置全局字体

# 数据集名称及对应的文件夹路径
datasets = {
    "lorenz": r"C:\Users\86178\Desktop\Chaotic Net\运行结果\预测结果\lorenz\长期时间信息提取能力",
    "power": r"C:\Users\86178\Desktop\Chaotic Net\运行结果\预测结果\power\长期时间信息提取能力"
}

# 时间窗口长度列表
window_sizes = ['8','16','24', '32', '40', '48', '56', '64']
# 存储不同数据集下，各窗口长度对应的评估指标
rmse_data = {ds: [] for ds in datasets}

# 模型列表及对应的样式配置（参考示例代码风格）
models = [
    "Chaotic-Net", "ModernTCN", "LSTM", "TCN",
    "Att_CNN_LSTM", "DLinear", "Transformer", "ESN", "WESN"
]

# 颜色配置（使用更美观的配色方案）
colors = [
    '#FF2300',  # 红色（突出显示Chaotic-Net）
    '#3490de',  # 深蓝色
    '#f07b3f',  # 橙色
    '#91eae4',  # 浅绿色
    '#38ef7d',  # 鲜亮绿
    '#f9ed69',  # 淡黄色
    '#bdef0a',  # 亮绿
    '#f948f7',  # 淡紫色
    '#954527'  # 棕色
]

# 标记样式配置
markers = ['D', 's', '^', 'o', 'v', 'p', '*', 'h', 'X']

# 透明度配置（突出显示主要模型）
alpha_dict = {
    "Chaotic-Net": 1.0,
    "ModernTCN": 1.0,
    "LSTM": 0.8,
    "TCN": 0.8,
    "Att_CNN_LSTM": 0.8,
    "DLinear": 0.8,
    "Transformer": 0.8,
    "ESN": 0.8,
    "WESN": 0.8
}


def get_all_result(test_y, y_pred):
    test_y = np.nan_to_num(test_y)
    y_pred = np.nan_to_num(y_pred)
    mse = mean_squared_error(test_y, y_pred)
    rmse = mean_squared_error(test_y, y_pred, squared=False)
    mae = mean_absolute_error(test_y, y_pred)
    mape = mean_absolute_percentage_error(test_y, y_pred)
    r2 = r2_score(test_y, y_pred)

    print(f'mse:{mse}, rmse:{rmse}, mae:{mae},mape:{mape},r2:{r2}')
    return mse, rmse, mae, mape, r2


# 读取数据并计算评估指标
for ds_name, folder_path in datasets.items():
    for ws in window_sizes:
        # 拼接不同模型的文件名
        file_paths = {
            "Chaotic-Net": os.path.join(folder_path, f"sr_{ds_name}_Chaotic-Net_({ws}_6).csv"),
            "ModernTCN": os.path.join(folder_path, f"tr_{ds_name}_ModernTCN_({ws}_6).csv"),
            "LSTM": os.path.join(folder_path, f"tr_{ds_name}_LSTM_({ws}_6).csv"),
            "TCN": os.path.join(folder_path, f"tr_{ds_name}_TCN_({ws}_6).csv"),
            "Att_CNN_LSTM": os.path.join(folder_path, f"tr_{ds_name}_Att_CNN_LSTM_({ws}_6).csv"),
            "DLinear": os.path.join(folder_path, f"tr_{ds_name}_DLinear_({ws}_6).csv"),
            "Transformer": os.path.join(folder_path, f"tr_{ds_name}_Transformer_({ws}_6).csv"),
            "ESN": os.path.join(folder_path, f"tr_{ds_name}_ESN_({ws}_6).csv"),
            "WESN": os.path.join(folder_path, f"tr_{ds_name}_WESN_({ws}_6).csv")
        }

        try:
            # 读取第一个模型的列名作为参考
            sample_df = pd.read_csv(file_paths["Chaotic-Net"])
            true_cols = [col for col in sample_df.columns if col.startswith('true')]
            pred_cols = [col for col in sample_df.columns if col.startswith('pred')]

            # 存储当前窗口大小的所有模型RMSE
            current_rmse = {"window_size": ws}

            # 计算各模型的RMSE
            for model in models:
                df = pd.read_csv(file_paths[model])
                print(ws,model)
                _, rmse, _, _, _ = get_all_result(
                    df[true_cols].values.flatten(),
                    df[pred_cols].values.flatten()
                )
                current_rmse[model] = rmse

            rmse_data[ds_name].append(current_rmse)

        except FileNotFoundError as e:
            print(f"文件 {e.filename} 未找到，跳过此时间窗口。")
            continue

print(rmse_data)

# 创建绘图
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True)

# 绘制主图
for i, (ds_name, data_list) in enumerate(rmse_data.items()):
    ax = axes[i]
    # 提取窗口大小（确保与数据长度匹配）
    plot_window_sizes = [d["window_size"] for d in data_list]

    # 绘制各模型曲线
    for idx, model in enumerate(models):
        rmse_values = [d[model] for d in data_list]
        ax.plot(plot_window_sizes, rmse_values,
                color=colors[idx],
                marker=markers[idx],
                linewidth=2,
                markersize=8,
                alpha=alpha_dict[model],
                label=model)

    # 设置子图标题和标签
    ax.set_title(f'{ds_name.capitalize()} Dataset', fontsize=16)
    ax.set_xlabel('Time Window Length', fontsize=15)
    ax.set_ylabel('RMSE', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.grid(True, linestyle='--', alpha=0.7)


# 创建自定义图例（统一放在图上方）
handles = [plt.Line2D([0], [0],
                      color=colors[i],
                      marker=markers[i],
                      linestyle='-',
                      markersize=8,
                      linewidth=2) for i in range(len(models))]

fig.legend(handles=handles,
           labels=models,
           loc='lower center',
           bbox_to_anchor=(0.5, 0.9),
           ncol=5,
           fontsize=13,
           frameon=False)

# 调整布局
plt.subplots_adjust(top=0.85, bottom=0.15, left=0.08, right=0.97, wspace=0.25)
# plt.savefig(r'C:\Users\86178\Desktop\Chaotic Net\运行结果\绘图\实验结果图\长时间信息提取.png',dpi=300)
plt.show()