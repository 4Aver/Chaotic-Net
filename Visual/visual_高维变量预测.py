import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import matplotlib.gridspec as gridspec

import warnings
from 研究生课题.测试方法.utils import get_all_result

warnings.filterwarnings('ignore')

plt.rcParams['font.size'] = 25
plt.rcParams['font.family'] = 'Times New Roman'

# 定义新的模型列表和路径
models = ['Att_CNN_LSTM', 'LSTM', 'ESN', 'TCN', 'ModernTCN', 'Transformer', 'DLinear', 'iChaotic-Net']
base_path = r'C:\Users\86178\Desktop\Chaotic Net\运行结果\预测结果\lorenz-96'


def get_max_min(type_name):
    data = pd.read_csv(r'C:\Users\86178\Desktop\Chaotic Net\Data\Original Data\{}.csv'.format(type_name)).iloc[:,0]
    data_max, data_min = data.max(), data.min()
    return data_max, data_min


def read_series_std_mean(type_name, model_name):
    list_seeds = [1, 2, 3, 4, 5]  # 新的随机种子
    list_res = []
    for seed in list_seeds:
        true, pred = read_data(type_name, model_name, seed, is_need_error=False, data_len=800)
        res = true - pred
        list_res.append(res)
    np_res = np.array(list_res)
    mean, std = np_res.mean(axis=0), np_res.std(axis=0)
    return mean, std


def read_data(type_name, model_name, seed=1, is_need_error=False, data_len=800):
    # 构建新的文件路径
    file_path = f"{base_path}\\tr_{type_name}_{model_name}_{seed}.csv"

    # 读取数据
    data = pd.read_csv(file_path).iloc[-data_len:, 1:]

    true, pred = data.loc[:, 'true'], data.loc[:, 'pred']
    print(model_name, seed, ':')
    get_all_result(true, pred)

    data_max, data_min = get_max_min(type_name)
    true = (true + 1) * (data_max - data_min) / 2 + data_min
    pred = (pred + 1) * (data_max - data_min) / 2 + data_min

    if is_need_error == False:
        return true, pred
    else:
        mse, rmse, mae, mape, r2 = get_all_result(true, pred, is_print=False)
        return rmse, mae, r2


def read_model_error(type_name, model_name, return_values=False):
    list_seeds = [1, 2, 3, 4, 5]  # 新的随机种子
    if return_values == False:
        list_rmse, list_mae, list_r2 = [], [], []
        for seed in list_seeds:
            rmse, mae, r2 = read_data(type_name, model_name, seed, is_need_error=True, data_len=800)
            list_rmse.append(rmse)
            list_mae.append(mae)
            list_r2.append(r2)
        print(model_name, 'mean list_rmse, list_mae, list_r2:', np.mean(list_rmse),np.mean(list_mae),np.mean(list_r2),)


        return list_rmse, list_mae, list_r2
    else:
        list_true, list_pred = [], []
        for seed in [1]:  # 使用第一个种子
            true, pred = read_data(type_name, model_name, seed, is_need_error=False, data_len=800)
            list_true.append(true)
            list_pred.append(pred)
        trues = np.concatenate(list_true, axis=0)
        preds = np.concatenate(list_pred, axis=0)
        return trues, preds


def make_new_plot(type_name):
    fig = plt.figure(figsize=(40, 45))

    # 定义3行3列的网格布局
    gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1])

    # ==========================================================================
    # 第一行：预测值对比图（占据整个第一行）
    # ==========================================================================
    ax_main = fig.add_subplot(gs[0, :])

    data_len = 300
    list_models = models  # 使用新的模型列表
    list_models_plot = ['LSTM','TCN','Transformer', 'ModernTCN', 'DLinear', 'iChaotic-Net']

    colors = ['#25E712', '#edf798', '#80d6ff', '#edb1f1','#f06868', '#3490de']

    # 为每个模型定义不同的点标记样式
    markers = ['o', 's', '^', 'D', 'v', '*', 'p', '>']
    marker_sizes = [18, 18, 18, 18, 18, 18, 18, 18]  # 不同标记的大小
    marker_every = 10  # 每隔多少点标记一次

    # 获取各模型预测数据
    true, pred_models = {}, {}
    for model in list_models:
        true[model], pred_models[model] = read_data(type_name, model, seed=1,data_len = 300)

    # 横坐标
    x_axis_data = list(range(data_len))

    # 绘制各模型预测线（点线图）
    for i, model in enumerate(list_models_plot):
        # 绘制线条
        ax_main.plot(x_axis_data, pred_models[model], color=colors[i],
                     label=model, linewidth=15)

        # 添加点标记
        ax_main.plot(x_axis_data[::marker_every],
                     pred_models[model][::marker_every],
                     color=colors[i],
                     marker=markers[i],
                     markersize=marker_sizes[i],
                     linestyle='none')

    # 绘制真实值
    ax_main.plot(x_axis_data, true[list_models[0]], color='black',
                 label='True', linewidth=10, linestyle='--')

    # 设置标签和标题
    if type_name == 'lorenz-96':
        ax_main.set_ylabel('Lorenz-96 x(t)', fontsize=40)

    ax_main.set_xlabel('t', fontsize=40)
    ax_main.legend(loc='upper right', fontsize=35)
    ax_main.grid(True, linestyle='--', linewidth=1)
    ax_main.tick_params(axis='both', which='major', labelsize=35)


    # ==========================================================================
    # 第二行和第三行的第一列：三个箱型图（垂直排列，占据两行高度）
    # ==========================================================================
    gs_box = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1:3, 0], hspace=0.3)

    list_model_metrics = [[], [], []]  # 存储RMSE, MAE, R2
    for model in list_models:
        list_rmse, list_mae, list_r2 = read_model_error(type_name, model)
        list_model_metrics[0].append(list_rmse)
        list_model_metrics[1].append(list_mae)
        list_model_metrics[2].append(list_r2)

    boxprops = dict(linewidth=3, color='black')
    flierprops = dict(marker='o', markersize=12, linestyle='none')
    medianprops = dict(linewidth=4, color='red')
    whiskerprops = dict(linewidth=3, color='black')

    # 绘制三个箱型图，垂直排列
    metrics = ['RMSE', 'MAE', 'R²']
    for i in range(3):
        ax_box = fig.add_subplot(gs_box[i])
        ax_box.boxplot(
            list_model_metrics[i],
            labels=list_models,
            boxprops=boxprops,
            flierprops=flierprops,
            medianprops=medianprops,
            whiskerprops=whiskerprops
        )
        ax_box.set_ylabel(metrics[i], fontsize=35)
        ax_box.tick_params(axis='x', labelrotation=30, labelsize=28)
        ax_box.grid(True, linestyle='--', alpha=0.7)
        ax_box.tick_params(axis='y', labelsize=30)

    # ==========================================================================
    # 第二行第二列和第三列：残差图（8个模型）
    # ==========================================================================
    residual_colors = ['#f06868', '#edf798', '#80d6ff', '#edb1f1', '#9896f1', '#3490de', '#abedd8', '#25E712']

    gs_residual = gridspec.GridSpecFromSubplotSpec(
        4, 2, subplot_spec=gs[1:3, 1:3], hspace=0.25, wspace=0.2
    )

    # 绘制所有8个残差图
    for idx, model in enumerate(list_models):
        ax_res = fig.add_subplot(gs_residual[idx // 2, idx % 2])
        mean, std = read_series_std_mean(type_name, model)
        time = np.array(range(1, 801))

        # 获取当前模型的颜色
        color = residual_colors[idx]

        # 绘制残差曲线
        ax_res.plot(time, mean, color=color, linewidth=3)
        ax_res.fill_between(time, mean - std, mean + std,
                            color=color, alpha=0.4)

        # 计算并显示统计值
        mean_residual = np.mean(mean)
        std_residual = np.std(mean)
        stats_text = f'Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}'
        ax_res.text(0.02, 0.85, stats_text, transform=ax_res.transAxes,
                    fontsize=22,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # 添加模型名称文本
        ax_res.text(0.5, 0.95, model, transform=ax_res.transAxes,
                    fontsize=32,
                    ha='center', va='top',
                    color='black',
                    weight='bold',
                    bbox=dict(
                        facecolor=color,
                        alpha=0.9,
                        edgecolor=color,
                        linewidth=3,
                        boxstyle='round,pad=0.2'
                    ))

        # 设置通用格式
        ax_res.set_xlabel('t', fontsize=25)
        ax_res.set_ylabel('Residual', fontsize=25)
        ax_res.grid(True, linestyle='--', alpha=0.6)
        ax_res.tick_params(labelsize=20)
        ax_res.set_ylim(-5, 5)


    # ==========================================================================
    # 添加圆角框和标签（核心修改部分）
    # ==========================================================================

    # 预测值图圆角框（红色）
    rect_pred = FancyBboxPatch(
        (0.001, 0.54), 0.98, 0.435,  # x, y, width, height (相对位置)
        boxstyle="round,pad=0.01,rounding_size=0.02",  # 圆角设置：rounding_size控制圆角大小
        linewidth=4, edgecolor='red', linestyle='-',
        facecolor='none', transform=fig.transFigure
    )
    fig.add_artist(rect_pred)

    # 预测值图标签(a)
    fig.text(0.97, 0.97, '(a)', fontsize=40, weight='bold',
             ha='right', va='top', color='red',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # 箱型图圆角框（绿色）
    rect_box = FancyBboxPatch(
        (0.001, 0.01), 0.323, 0.495,  # x, y, width, height (相对位置)
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=4, edgecolor='green', linestyle='-',
        facecolor='none', transform=fig.transFigure
    )
    fig.add_artist(rect_box)

    # 箱型图标签(b)
    fig.text(0.32, 0.515, '(b)', fontsize=40, weight='bold',
             ha='right', va='top', color='green',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # 残差图圆角框（蓝色）
    rect_res = FancyBboxPatch(
        (0.365, 0.01), 0.616, 0.495,  # x, y, width, height (相对位置)
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=4, edgecolor='blue', linestyle='-',
        facecolor='none', transform=fig.transFigure
    )
    fig.add_artist(rect_res)

    # 残差图标签(c)
    fig.text(0.98, 0.515, '(c)', fontsize=40, weight='bold',
             ha='right', va='top', color='blue',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # 调整整体布局
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.98)
    plt.savefig(r'C:\Users\86178\Desktop\Chaotic Net\运行结果\绘图\实验结果图\{}.png'.format(type_name),
                dpi=300, bbox_inches='tight')
    # plt.show()


# 调用函数生成图表
make_new_plot('lorenz-96')