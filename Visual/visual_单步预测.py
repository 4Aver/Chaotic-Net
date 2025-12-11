import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from 对比实验.utils import get_all_result
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import ScalarFormatter
from 研究生课题.测试方法.utils import get_all_result
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import warnings
warnings.filterwarnings('ignore')


def get_max_min(type_name):
    data = pd.read_csv(r'C:\Users\86178\Desktop\Chaotic Net\Data\Original Data\{}.csv'.format(type_name))

    if type_name == 'power':
        target_col = data.iloc[:,3]
        data_max, data_min = target_col.max(), target_col.min()
    else:
        target_col = data.iloc[:,0]
        data_max, data_min = target_col.max(), target_col.min()
    return data_max, data_min


def read_data(type_name, model_name,train_number,return_metric=False):
    data_len = 800

    if model_name == 'Chaotic-Net':
        load_name = 'sr'
    else:
        load_name = 'tr'

    data = pd.read_csv(
        r'C:\Users\86178\Desktop\Chaotic Net\运行结果\预测结果\{}\多次实验\{}_{}_{}_{}.csv'.format(type_name, load_name, type_name,
                                                                                  model_name,train_number)).iloc[-data_len:, 1:]

    true, pred = data.loc[:, 'true'], data.loc[:, 'pred']
    data_max, data_min = get_max_min(type_name)
    true = np.array((true + 1) * (data_max - data_min) / 2 + data_min)
    pred = np.array((pred + 1) * (data_max - data_min) / 2 + data_min)
    residuals = [true - predicted for true, predicted in zip(true, pred)]

    if return_metric:
        mse, rmse, mae, mape, r2 = get_all_result(true, pred,is_print=True)
        return rmse, mae, r2
    else:
        return true, pred, residuals


def make_type(type_name):
    models = ['Att_CNN_LSTM','LSTM','ESN','WESN','TCN','ModernTCN','Transformer','DLinear','Chaotic-Net']

    min_ylim, max_ylim = 0, 0
    if type_name == 'lorenz':
        min_ylim,max_ylim = -0.2, 0.2
    if type_name == 'rossler':
        min_ylim,max_ylim = -0.2, 0.2
    if type_name == 'power':
        min_ylim, max_ylim = -4000, 4000

    residuals = [read_data(type_name, model,train_number=2)[2] for model in models]
    min_residual = min([np.min(res) for res in residuals])
    max_residual = max([np.max(res) for res in residuals])
    bins = np.linspace(min_residual, max_residual, 50)

    # 创建一个新的gridspec布局
    # fig = plt.figure(figsize=(25, 40))  # 调整整体画布大小
    fig = plt.figure(figsize=(30, 45))  # 调整整体画布大小

    # 创建gridspec
    gs = gridspec.GridSpec(nrows=9, ncols=5, figure=fig)

    for i, model in enumerate(models):
        print('model name:',model)
        true, pred, res = read_data(type_name, model, train_number=4)  # 获取数据

        list_data = [[],[],[]]
        for train_number in range(1,6):
            rmse, mae, r2 = read_data(type_name, model, train_number=train_number,return_metric=True)  # 获取数据
            list_data[0].append(rmse)
            list_data[1].append(mae)
            list_data[2].append(r2)

        rmse_array = np.array(list_data[0])
        mae_array = np.array(list_data[1])
        r2_array = np.array(list_data[2])

        rmse_mean = rmse_array.mean()
        rmse_std = rmse_array.std()

        mae_mean = mae_array.mean()
        mae_std = mae_array.std()

        r2_mean = r2_array.mean()
        r2_std = r2_array.std()

        print(f"RMSE: 均值 = {rmse_mean:.5f}, 标准差 = {rmse_std:.8f}")
        print(f"MAE:  均值 = {mae_mean:.5f}, 标准差 = {mae_std:.8f}")
        print(f"R2:   均值 = {r2_mean:.8f}, 标准差 = {r2_std:.8f}")

        # 为真实值和预测值图创建更大的空间，合并两列
        ax0 = fig.add_subplot(gs[i, :2])
        # 对其他图保持原有的单列设置
        ax1 = fig.add_subplot(gs[i, 2])
        ax2 = fig.add_subplot(gs[i, 3])
        ax3 = fig.add_subplot(gs[i, 4])
        ax3.set_xticks([])
        ax3.set_yticks([])

        # 绘制真实值和预测值的线图，注意第一列和第二列
        ax0.plot(true, label='True Values', color='red')
        ax0.plot(pred, 'b--', label='Predicted Values')
        ax0.tick_params(axis='y', direction='in')

        ax0_inset = inset_axes(ax0,
                             width="28%",  # 子图宽度比例
                             height="70%",  # 子图高度比例
                             loc='upper right',  # 初始位置
                             borderpad=1)  # 边距

        zoom_start, zoom_end = 0, 10

        if type_name == 'lorenz':
            zoom_start, zoom_end = 385, 395
        if type_name == 'rossler':
            zoom_start, zoom_end = 255, 275
        if type_name == 'power':
            zoom_start, zoom_end = 360, 390
        if type_name == 'lorenz_96':
            zoom_start, zoom_end = 350, 380

        ax0_inset.plot(range(zoom_start, zoom_end + 1),true[zoom_start:zoom_end + 1], color='red', lw=2.5)
        ax0_inset.plot(range(zoom_start, zoom_end + 1),pred[zoom_start:zoom_end + 1], 'b--', lw=2.5)
        ax0_inset.grid()
        ax0_inset.set_yticks([])
        ax0_inset.tick_params(axis='x',direction='in', labelsize=14)
        ax0_inset.tick_params(axis='y', direction='in')

        ax0.add_patch(
            plt.Rectangle((zoom_start, min(true.min(), pred.min())),
                          zoom_end - zoom_start,
                          max(true.max(), pred.max()) - min(true.min(), pred.min()),
                          fill=False,
                          ls='--',
                          edgecolor='red',
                          lw=4))

        # 添加文本框，用橙色框住并填充
        ax0.text(0.08, 1, model, transform=ax0.transAxes, fontsize=25, verticalalignment='top',
                 horizontalalignment='center',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', edgecolor='orange', alpha=1))

        # 绘制残差的直方图
        ax1.hist(res, bins=bins, density=True, color=tuple(np.array([0.2, 0.2, 0.8]) + 0.2), edgecolor='black')
        mean = np.mean(res)
        variance = np.var(res)
        x = np.linspace(mean - 3 * np.sqrt(variance), mean + 3 * np.sqrt(variance), 100)
        ax1.plot(x, stats.norm.pdf(x, mean, np.sqrt(variance)), color='red', linestyle='--')

        if type_name == 'power':
            stats_text = (f"$\mu$: {mean:.5f}\n"
                          f"$\sigma$: {variance**0.5:.5f}")

            ax1.text(0.05, 0.9, stats_text, transform=ax1.transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=20
                       )

            # ax1.text(mean*0.7, max(ax1.get_ylim()), "$\mu$ = {:.2f}\n$\sigma^2$ = {:.2f}".format(mean,variance ** 0.5), ha='right', va='top', fontsize=23)
            # ax1.text(mean*0.7, max(ax1.get_ylim()) * 0.8, r"$\sigma^2$ = {:.2f}".format(variance ** 0.5), ha='right',
            #          va='top', fontsize=23)

            # 添加科学计数法格式设置
            formatter = ScalarFormatter(useMathText=True)  # 使用数学文本格式
            formatter.set_scientific(True)  # 开启科学计数法
            formatter.set_powerlimits((-1, 1))  # 设置科学计数法的阈值
            # ax0.yaxis.set_major_formatter(formatter)  # 应用到Y轴
            ax1.yaxis.set_major_formatter(formatter)  # 应用到Y轴
            ax2.yaxis.set_major_formatter(formatter)  # 应用到Y轴

            ax0.set_yticks([20000, 40000])
            ax0.tick_params(axis='y', which='major', rotation=50)

            # ax0.set_ylabel('Values(KWh)', fontsize=17)

        else:
            ax1.text(mean*0.7, max(ax1.get_ylim()), "$\mu$ = {:.2f}\n$\sigma^2$ = {:.2f}".format(mean,variance ** 0.5), ha='right', va='top', fontsize=23)

            # ax1.text(mean*0.7, max(ax1.get_ylim()), r"$\mu$ = {:.5f}".format(mean), ha='right', va='top', fontsize=23)
            # ax1.text(mean*0.7, max(ax1.get_ylim()) * 0.8, r"$\sigma^2$ = {:.5f}".format(variance ** 0.5), ha='right',
            #          va='top', fontsize=23)
        # ax1.set_xlabel('Prediction Errors', fontsize=17)
        # ax1.set_ylabel('Probability Density', fontsize=17)
        ax1.tick_params(axis='y', direction='in')

        # 绘制残差线图
        ax2.plot(res, color='purple')
        ax2.axhline(y=0, color='red', linestyle='--')
        ax2.set_ylim(min_ylim,max_ylim)
        # ax2.set_xlabel('True Values',fontsize=17)
        # ax2.set_ylabel('Residuals',fontsize=17)
        ax2.tick_params(axis='y', direction='in')

        # 创建4个新的子图
        # gs_ax3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=ax3, wspace=1.9)
        gs_ax3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[i, 4], wspace=1.9)

        ax3_sub1 = fig.add_subplot(gs_ax3[0])
        ax3_sub2 = fig.add_subplot(gs_ax3[1])
        # ax3_sub3 = fig.add_subplot(gs_ax3[2])

        # 绘制箱型图
        sns.boxplot(data=list_data[0], ax=ax3_sub1, color='#4EAB90')
        # sns.stripplot(data=list_data[0], ax=ax3_sub1, color='red', alpha=0.5)
        # ax3_sub1.set_title('RMSE',fontsize=17)
        ax3_sub1.tick_params(axis='y', direction='in',labelsize=22)
        ax3_sub1.tick_params(axis='x', labelbottom=False, direction='in')
        ax3_sub1.yaxis.set_ticks_position('right')

        sns.boxplot(data=list_data[1], ax=ax3_sub2, color='#8EB69C')
        # sns.stripplot(data=list_data[1], ax=ax3_sub2, color='red', alpha=0.5)
        # ax3_sub2.set_title('MAE',fontsize=17)
        ax3_sub2.tick_params(axis='y', direction='in',labelsize=22)
        ax3_sub2.tick_params(axis='x', labelbottom=False, direction='in')

        # sns.boxplot(data=list_data[2], ax=ax3_sub3, color='#EDDDC3')
        # # sns.stripplot(data=list_data[2], ax=ax3_sub3, color='red', alpha=0.5)
        # # ax3_sub3.set_title('Run Time (s)',fontsize=17)
        # ax3_sub3.tick_params(axis='y', direction='in',labelsize=18)
        # ax3_sub3.tick_params(axis='x', labelbottom=False, direction='in')
        # # ax3_sub3.yaxis.set_ticks_position('right')

        # 设置Y轴刻度标签格式
        if type_name != 'power':
            def format_y_tick(value, pos):
                return "{:.3f}".format(value)

            formatter = FuncFormatter(format_y_tick)
            for ax in [ax3_sub2]:
                ax.yaxis.set_major_formatter(formatter)

        # 调整子图刻度字体大小
        ax0.tick_params(axis='x', labelsize=23)
        ax0.tick_params(axis='y', labelsize=20 if type_name=='power' else 23)

        ax1.tick_params(axis='both', labelsize=23)
        ax2.tick_params(axis='both', labelsize=23)

        # if i == 0:
        #     ax1.set_title('Distribution of Prediction Errors', fontsize=17)
        #     ax1.get_yaxis().get_offset_text().set_x(-0.15)  # 调整科学技术法值的位置
        #     ax2.set_title('Residual Plot', fontsize=17)

    # for i in range(9):
    #     if i%2 == 0:
    #         color = '#f948f7'
    #     else:
    #         color = '#c76813'
    #     red_rect = patches.Rectangle((0.004, 0.027 + 0.081*i), 0.99, 0.078, linewidth=3, edgecolor=color,linestyle='--',facecolor='none', transform=fig.transFigure, clip_on=False)
    #     fig.add_artist(red_rect)

    # red_rect = patches.Rectangle((0.004, 0.108), 0.99, 0.078, linewidth=3, edgecolor='blue',linestyle='--',facecolor='none', transform=fig.transFigure, clip_on=False)
    # fig.add_artist(red_rect)

    # 调整子图间距
    plt.subplots_adjust(left=0.03, bottom=0.04, right=0.98, top=0.99,hspace=0.35, wspace=0.2)

    red_rect = patches.Rectangle((0.05, 0.005), 0.935, 0.02, linewidth=3, edgecolor='green',linestyle='--',facecolor='none', transform=fig.transFigure, clip_on=False)
    fig.add_artist(red_rect)

    legend_elements = [
        plt.Line2D([0], [0], color='red', lw=2, linestyle='-', label='True Values'),
        plt.Line2D([0], [0], color='blue', lw=2, linestyle='--', label='Predicted Values')
    ]
    # 添加全局图例
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.15, 0.0001),  # 调整位置（x,y坐标）
        ncol=1,
        fontsize=23,
        frameon=False
    )

    # 添加坐标轴说明文本（调整到图例上方）
    fig.text(
        x=0.23,
        y=0.005,  # 根据实际效果微调
        s="X axis - Sample \nY axis - Value (KWh)" if type_name=='power' else "X axis - Sample \nY axis - Value" ,
        va='bottom',
        fontsize=25,
        color='red',
        transform=fig.transFigure
    )

    fig.text(
        x=0.43,
        y=0.005,  # 根据实际效果微调
        s="X axis - Prediction Errors \nY axis - Probability Density",
        va='bottom',
        fontsize=25,
        color='red',
        transform=fig.transFigure
    )

    fig.text(
        x=0.65,
        y=0.005,  # 根据实际效果微调
        s="X axis - Sample \nY axis - Residuals",
        va='bottom',
        fontsize=25,
        color='red',
        transform=fig.transFigure
    )

    fig.text(
        x=0.85,
        y=0.005,  # 根据实际效果微调
        s="X axis - RMSE  MAE \nY axis - Values",
        va='bottom',
        fontsize=25,
        color='red',
        transform=fig.transFigure
    )

    # 最终调整和保存
    plt.savefig(r'C:\Users\86178\Desktop\Chaotic Net\运行结果\绘图\实验结果图\低分辨率_{}.png'.format(type_name),dpi=300)
    # plt.show()


make_type('lorenz')
# make_type('rossler')
# make_type('power')
# make_type('lorenz_96')
