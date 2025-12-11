import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# 设置英文字体为Times New Roman
plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
warnings.filterwarnings('ignore')


# 假设get_all_result函数已正确实现，这里使用实际逻辑
def get_all_result(true, pred, is_print=False):
    """计算评估指标"""
    mse = np.mean((true - pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(true - pred))
    mape = np.mean(np.abs((true - pred) / true)) * 100 if np.any(true != 0) else 0
    r2 = 1 - np.sum((true - pred) ** 2) / np.sum((true - np.mean(true)) ** 2)

    if is_print:
        print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    return mse, rmse, mae, mape, r2


def get_max_min(type_name):
    """获取数据的最大值和最小值用于反归一化"""
    # 实际数据读取逻辑
    data = pd.read_csv(r'C:\Users\86178\Desktop\Chaotic Net\Data\Original Data\{}.csv'.format(type_name))

    if type_name == 'power':
        target_col = data.iloc[:, 3]
    else:
        target_col = data.iloc[:, 0]

    return target_col.max(), target_col.min()


def read_data(type_name, load_name, parameter, is_kernel=True):
    """读取数据并计算评估指标"""
    data_len = 800

    # 根据参数类型选择正确的文件路径
    if is_kernel:
        file_path = r'C:\Users\86178\Desktop\Chaotic Net\运行结果\预测结果\超参实验\{}_{}_Chaotic-Net_1_DMCM_kernel_{}.csv'.format(
            load_name, type_name, parameter)
    else:
        file_path = r'C:\Users\86178\Desktop\Chaotic Net\运行结果\预测结果\超参实验\{}_{}_Chaotic-Net_1_D_{}.csv'.format(
            load_name, type_name, parameter)

    # 读取实际数据
    data = pd.read_csv(file_path).iloc[-data_len:, 1:]
    true, pred = data.loc[:, 'true'], data.loc[:, 'pred']

    # 反归一化
    data_max, data_min = get_max_min(type_name)
    true = np.array((true + 1) * (data_max - data_min) / 2 + data_min)
    pred = np.array((pred + 1) * (data_max - data_min) / 2 + data_min)

    # 计算评估指标
    mse, rmse, mae, mape, r2 = get_all_result(true, pred, is_print=False)
    return rmse, mae, r2


def plot_parameter_comparison():
    """绘制参数对比图表"""
    # 定义参数、数据集和对应的背景颜色
    kernel_params = [1, 3, 5, 7, 9]
    d_params = [8, 16, 32, 64, 128]
    datasets = ['lorenz', 'rossler', 'power']
    background_colors = ['#f06868', '#edf798', '#80d6ff']  # 对应三个数据集的背景色
    load_names = ['sr', 'tr', 'sr']  # 对应每个数据集的load_name

    # 收集数据
    kernel_data = {dataset: {'rmse': [], 'mae': []} for dataset in datasets}
    d_data = {dataset: {'rmse': [], 'mae': []} for dataset in datasets}

    # 收集kernel参数的数据
    for i, dataset in enumerate(datasets):
        for kernel in kernel_params:
            rmse, mae, _ = read_data(dataset, load_names[i], kernel, is_kernel=True)
            kernel_data[dataset]['rmse'].append(rmse)
            kernel_data[dataset]['mae'].append(mae)

    # 收集d参数的数据
    for i, dataset in enumerate(datasets):
        for d in d_params:
            rmse, mae, _ = read_data(dataset, load_names[i], d, is_kernel=False)
            d_data[dataset]['rmse'].append(rmse)
            d_data[dataset]['mae'].append(mae)

    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))

    # 绘制第一行：d参数（原第二行）
    for col, dataset in enumerate(datasets):
        ax = axes[0, col]
        ax.plot(d_params, d_data[dataset]['rmse'], 'o-', label='RMSE', color='#1f77b4', linewidth=2)
        ax.plot(d_params, d_data[dataset]['mae'], 's-', label='MAE', color='#ff7f0e', linewidth=2)

        # 在右上角添加带彩色背景的数据集文本标识
        ax.text(0.95, 0.95, dataset.capitalize(), transform=ax.transAxes,
                fontsize=10, fontweight='bold', ha='right', va='top',
                bbox=dict(facecolor=background_colors[col], edgecolor='gray', alpha=0.7))

        ax.set_xlabel('D', fontsize=10)
        ax.set_xticks(d_params)
        ax.grid(alpha=0.3)
        ax.legend()

    # 绘制第二行：kernel参数（原第一行）
    for col, dataset in enumerate(datasets):
        ax = axes[1, col]
        ax.plot(kernel_params, kernel_data[dataset]['rmse'], 'o-', label='RMSE', color='#1f77b4', linewidth=2)
        ax.plot(kernel_params, kernel_data[dataset]['mae'], 's-', label='MAE', color='#ff7f0e', linewidth=2)

        # 在右上角添加带彩色背景的数据集文本标识
        ax.text(0.95, 0.95, dataset.capitalize(), transform=ax.transAxes,
                fontsize=10, fontweight='bold', ha='right', va='top',
                bbox=dict(facecolor=background_colors[col], edgecolor='gray', alpha=0.7))

        ax.set_xlabel('Kernel', fontsize=10)
        ax.set_xticks(kernel_params)
        ax.grid(alpha=0.3)
        ax.legend()

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.983, left=0.051, bottom=0.069, right=0.986, hspace=0.27, wspace=0.175)
    # plt.show()
    plt.savefig(r'C:\Users\86178\Desktop\Chaotic Net\运行结果\绘图\实验结果图\超参数实验.png',dpi=300)

    return fig


# 运行绘图函数
if __name__ == "__main__":
    plot_parameter_comparison()
