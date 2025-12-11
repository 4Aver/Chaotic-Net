import torch
import numpy as np
import pandas as pd
from io import BytesIO
from nolds import lyap_e
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import gaussian_kde
# 注意：请根据实际情况调整导入路径
from 研究生课题.测试方法.utils import get_all_result

import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10


def combined_ks_visualization(t_index=100, D=80):
    # 模型名称列表
    model_names = ['LSTM', 'TCN', 'ESN', 'Transformer', 'Att_CNN_LSTM', 'DLinear', 'ModernTCN', 'iChaotic-Net']
    n_models = len(model_names)  # 模型数量

    # 加载数据
    models_data = {}
    all_errors = []  # 收集所有模型的误差数据
    for model in model_names:
        # 替换为您的实际数据路径
        path = rf'C:\Users\86178\Desktop\Chaotic Net\运行结果\预测结果\KS\tr_KS_{model}.csv'
        try:
            data = pd.read_csv(path)
            true = np.array(data.loc[:, 'true1':'true80'])
            pred = np.array(data.loc[:, 'pred1':'pred80'])

            print(f"已加载 {model} 数据，形状: {true.shape}")
            get_all_result(true, pred)

            # 计算误差
            error = true - pred  # 绝对误差
            all_errors.append(error.flatten())  # 收集所有误差用于确定全局范围

            models_data[model] = {
                'true': true,
                'pred': pred,
                'error': error
            }
        except Exception as e:
            print(f"加载 {model} 失败: {e}")

    # 固定颜色范围（根据实际数据调整）
    VMIN, VMAX = 0.0, 0.02

    # 计算全局误差范围以固定坐标轴
    all_errors_flattened = np.concatenate(all_errors)
    global_min = np.min(all_errors_flattened)
    global_max = np.max(all_errors_flattened)
    density_max = 0
    for errors in all_errors:
        hist, bin_edges = np.histogram(errors, bins=100, density=True)
        density_max = max(density_max, hist.max())

    # 添加10%的余量
    hist_x_min = -1
    hist_x_max = 1
    hist_y_max = density_max * 1.2  # 20%余量

    print(f"全局误差范围: [{global_min:.5f}, {global_max:.5f}]")
    print(f"直方图固定范围: X轴[{hist_x_min:.5f}, {hist_x_max:.5f}], Y轴[0, {hist_y_max:.5f}]")

    # 创建图形
    fig = plt.figure(figsize=(18, 3 * n_models))
    gs_master = GridSpec(n_models, 2, width_ratios=[1, 1], wspace=0.3)  # 主网格：n行2列

    # 颜色设置
    true_color = '#1f77b4'
    pred_color = '#ff7f0e'
    error_color = tuple(np.array([0.2, 0.2, 0.8]) + 0.2)  # 深蓝色
    norm_color = 'red'  # 正态分布曲线颜色
    kde_color = 'green'  # 核密度估计曲线颜色
    model_label_color = '#FFA500'  # 橙色背景
    # 为a/b/c/d设置不同颜色
    abc_color = {
        'a': '#1f77b4',    # 蓝色
        'b': '#ff7f0e',    # 橙色
        'c': '#2ca02c',    # 绿色
        'd': '#d62728'     # 红色
    }

    # 存储每个模型的轴对象，用于绘制边框
    axes_list = []

    for idx, (model_name, data) in enumerate(models_data.items()):
        # 提取数据
        true = data['true']
        pred = data['pred']
        error = data['error']
        all_errors = error.flatten()  # 所有误差的一维数组

        # 左侧区域：分为上下两部分
        gs_left = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_master[idx, 0], height_ratios=[1, 1], hspace=0.4)

        # 左上：折线图
        ax1 = fig.add_subplot(gs_left[0])

        # 左下：分为两个子图（2:1比例）
        gs_bottom = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_left[1], width_ratios=[2, 1], wspace=0.3)
        ax2_1 = fig.add_subplot(gs_bottom[0])  # 残差直方图（比例2）
        ax2_2 = fig.add_subplot(gs_bottom[1])  # 真实值预测值对比（比例1）

        # 右侧：误差热力图
        ax3 = fig.add_subplot(gs_master[idx, 1])

        # 存储当前模型的所有轴对象
        axes_list.append({
            'axes': [ax1, ax2_1, ax2_2, ax3],
            'name': model_name
        })

        # 1. 左上：空间模式折线图 - (a)标记（蓝色）
        ax1.plot(range(D), true[t_index], color=true_color, lw=2, label='True')
        ax1.plot(range(D), pred[t_index], color=pred_color, ls='--', lw=1.5, label='Predicted')
        ax1.text(-0.01, 1.1, '(a)', transform=ax1.transAxes,
                 fontsize=12, fontweight='bold', va='top', ha='right',
                 color=abc_color['a'])  # 添加颜色
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3, ls=':')
        ax1.set_xlim(0, D - 1)

        # 仅在第一个模型添加图例
        if idx == 0:
            true_patch = mpatches.Patch(color=true_color, label='True')
            pred_patch = mpatches.Patch(color=pred_color, linestyle='--', label='Predicted')
            ax1.legend(handles=[true_patch, pred_patch], loc='upper right')

        # 2.1 左下左侧：残差分布直方图 - (b)标记（橙色）
        res = all_errors  # 所有残差的一维数组

        # 计算统计量
        mean = np.round(np.mean(res), 5)
        std_dev = np.round(np.std(res), 5)
        variance = np.round(std_dev ** 2, 5)  # 方差
        median = np.round(np.median(res), 5)

        # 计算合适的bins范围（使用IQR方法）
        q25, q75 = np.percentile(res, [25, 75])
        iqr = q75 - q25
        bin_width = 2 * iqr * len(res) ** (-1 / 3)  # Freedman-Diaconis规则
        bins = np.arange(res.min(), res.max() + bin_width, bin_width)

        # 如果bins太少，强制最小数量
        if len(bins) < 10:
            bins = np.linspace(res.min(), res.max(), 30)

        # 绘制直方图（频率分布）
        n, bins, patches = ax2_1.hist(
            res,
            bins=bins,
            density=True,  # 使用密度模式而非频率
            color=error_color,
            edgecolor='black',
            alpha=0.7,
            orientation='vertical'
        )

        # 绘制正态分布曲线（基于均值和标准差）
        x_range = np.linspace(res.min(), res.max(), 200)
        norm_dist = stats.norm.pdf(x_range, mean, std_dev)
        ax2_1.plot(x_range, norm_dist, color=norm_color, linestyle='--', lw=2)

        # 固定直方图的X轴和Y轴范围
        ax2_1.set_xlim(hist_x_min, hist_x_max)
        ax2_1.set_ylim(0, hist_y_max)

        # 添加统计信息文本
        stats_text = (f"Mean: {mean:.5f}\n"
                      f"Variance: {variance:.5f}")

        ax2_1.text(0.05, 0.9, stats_text, transform=ax2_1.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                   )

        # 添加网格和标签，添加子图标记 (b)
        ax2_1.text(-0.01, 1.1, '(b)', transform=ax2_1.transAxes,
                   fontsize=12, fontweight='bold', va='top', ha='right',
                   color=abc_color['b'])  # 添加颜色
        ax2_1.grid(True, alpha=0.5, ls=':')
        ax2_1.set_xlabel('Prediction Error')
        ax2_1.set_ylabel('Density')
        # ax2_1.legend(loc='upper left', fontsize=8)

        # 2.2 左下右侧：真实值-预测值对比图 - (c)标记（绿色）
        if len(true.flatten()) > 10000:
            indices = np.random.choice(len(true.flatten()), 10000, replace=False)
            true_flat = true.flatten()[indices]
            pred_flat = pred.flatten()[indices]
        else:
            true_flat = true.flatten()
            pred_flat = pred.flatten()

        # 绘制散点图
        ax2_2.scatter(true_flat, pred_flat, alpha=0.5, s=5, color=pred_color, label='Prediction')

        # 确定坐标范围
        min_val = min(true_flat.min(), pred_flat.min())
        max_val = max(true_flat.max(), pred_flat.max())
        padding = (max_val - min_val) * 0.1
        min_range = min_val - padding
        max_range = max_val + padding

        # 绘制理论线（y=x）
        ax2_2.plot([min_range, max_range], [min_range, max_range], 'k--', lw=1.5, label='Theoretical')

        # 设置对称坐标轴范围
        ax2_2.set_xlim(min_range, max_range)
        ax2_2.set_ylim(min_range, max_range)

        # 添加标签和网格，添加子图标记 (c)
        ax2_2.text(-0.1, 1.1, '(c)', transform=ax2_2.transAxes,
                   fontsize=12, fontweight='bold', va='top', ha='right',
                   color=abc_color['c'])  # 添加颜色
        ax2_2.set_xlabel('Test Value')
        ax2_2.set_ylabel('Prediction')
        ax2_2.grid(True, alpha=0.3, ls=':')
        if idx == 0:
            ax2_2.legend(loc='upper left')


        # 3. 右侧：误差热力图 - (d)标记（红色）
        X, T = np.meshgrid(range(D), range(len(error)))
        im = ax3.pcolormesh(X, T, error, cmap='jet', shading='auto', vmin=VMIN, vmax=VMAX)
        ax3.text(-0.01, 1.05, '(d)', transform=ax3.transAxes,
                 fontsize=12, fontweight='bold', va='top', ha='right',
                 color=abc_color['d'])  # 添加颜色
        ax3.set_xlabel('Spatial Dimension')
        ax3.set_ylabel('Time Step')

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax3, pad=0.02)
        cbar.set_label('Error')
        cbar.set_ticks(np.linspace(VMIN, VMAX, 5))
        cbar.ax.set_yticklabels([f'{v:.4f}' for v in np.linspace(VMIN, VMAX, 5)])

        # 为最后一行添加x轴标签
        if idx == n_models - 1:
            ax1.set_xlabel('Spatial Dimension')
            ax2_1.set_xlabel('Prediction Error')
            ax2_2.set_xlabel('Test Value')
            ax3.set_xlabel('Spatial Dimension')

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.06, hspace=0.5)

    # 为每个模型添加**圆角**虚线框和带橙色背景的模型名称
    for model_idx, model_info in enumerate(axes_list):
        axes = model_info['axes']
        model_name = model_info['name']

        # 获取当前模型所有子图的位置信息
        bboxes = [ax.get_position() for ax in axes]

        # 计算模型区域的整体边界
        left = min(bbox.x0 for bbox in bboxes)
        right = max(bbox.x1 for bbox in bboxes)
        bottom = min(bbox.y0 for bbox in bboxes)
        top = max(bbox.y1 for bbox in bboxes)

        # 添加适当边距
        margin = 0.015
        left -= margin
        right += margin
        bottom -= margin
        top += margin

        # 绘制**圆角**虚线框（使用mpatches.FancyBboxPatch实现圆角）
        rect = mpatches.FancyBboxPatch(
            (left-0.015, bottom+0.007),
            right - left + 0.05,
            top - bottom - 0.025,
            boxstyle="round,pad=0.01,rounding_size=0.02",  # 圆角设置
            fill=False,
            edgecolor='black',
            linestyle='--',
            linewidth=1.5
        )
        fig.add_artist(rect)

        # 添加带橙色背景的模型名称
        text_x = left + (right - left) / 2  # 水平居中
        text_y = top - 0.1  # 框上方一点点

        plt.text(
            text_x+0.016,
            text_y+0.045,
            model_name,
            horizontalalignment='center',
            verticalalignment='bottom',
            transform=fig.transFigure,
            fontsize=12,
            fontweight='bold',
            bbox=dict(facecolor=model_label_color, alpha=0.7, boxstyle='round,pad=0.3')
        )

    # 保存图像
    save_path = r'C:\Users\86178\Desktop\Chaotic Net\运行结果\绘图\实验结果图\高维数据预测_KS_new.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图像已保存至: {save_path}")

    # plt.show()


if __name__ == "__main__":
    combined_ks_visualization(t_index=100, D=80)