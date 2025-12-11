import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from 研究生课题.测试方法.utils import get_all_result

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False


def calculate_r2(true, pred):
    """计算R²分数"""
    if len(true.shape) > 1 and true.shape[1] > 1:
        r2_scores = []
        for i in range(true.shape[1]):
            true_i = true.iloc[:, i].values
            pred_i = pred.iloc[:, i].values
            ss_res = np.sum((true_i - pred_i) ** 2)
            ss_tot = np.sum((true_i - np.mean(true_i)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
            r2_scores.append(r2)
        return np.mean(r2_scores)
    else:
        true_vals = true.values.flatten()
        pred_vals = pred.values.flatten()
        ss_res = np.sum((true_vals - pred_vals) ** 2)
        ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0


def make_r2_mae_scatter_plot():
    # ========== 核心配置（最终精准R²范围） ==========
    list_models = ['Att_CNN_LSTM', 'LSTM', 'ESN', 'WESN', 'TCN', 'ModernTCN', 'Transformer', 'DLinear', 'Chaotic-Net']
    datasets = ['lorenz', 'rossler', 'power']
    pred_lens = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']

    # 模型颜色（9个模型精准匹配）
    colors = [
        '#91eae4', '#38ef7d', '#f9ed69', '#bdef0a', '#3490de',
        '#00ff00', '#f07b3f', '#f948f7', '#954527'
    ]
    assert len(colors) == len(list_models), f"颜色数量({len(colors)})需匹配模型数量({len(list_models)})"

    # 【最终精准】每个数据集的R²范围
    axis_ranges = {
        'lorenz': {'r2': (0.999, 1.00001), 'mae': (0, None)},  # lorenz：0.9992~1.0005
        'rossler': {'r2': (0.9992, 1.00001), 'mae': (0, None)}, # rossler：0.9992~1.0005
        'power': {'r2': (0.92, 0.99), 'mae': (0, None)}        # power：0.92~0.99
    }

    # ========== 数据读取与指标计算 ==========
    def read_data(type_name):
        model_metrics = {model: {'mae': [], 'r2': []} for model in list_models}
        for model in list_models:
            load_name = 'tr' if model != 'Chaotic-Net' else ('tr' if type_name == 'rossler' else 'sr')
            for plen in pred_lens:
                try:
                    if plen == '1':
                        file_path = fr'C:\Users\86178\Desktop\Chaotic Net\运行结果\预测结果\{type_name}\多次实验\{load_name}_{type_name}_{model}_5.csv'
                    else:
                        file_path = fr'C:\Users\86178\Desktop\Chaotic Net\运行结果\预测结果\{type_name}\多步预测\{load_name}_{type_name}_{model}_42_({plen}).csv'
                    data = pd.read_csv(file_path).iloc[-800:, :]
                    true = data.filter(like='true')
                    pred = data.filter(like='pred')
                    _, rmse, mae, _, _ = get_all_result(true, pred)
                    r2 = calculate_r2(true, pred)
                    model_metrics[model]['mae'].append(mae)
                    model_metrics[model]['r2'].append(r2)
                except Exception as e:
                    print(f"警告：{type_name}-{model}-步数{plen} 出错: {e}")
                    continue
        # 计算均值和方差
        result = {}
        for model in list_models:
            metrics = model_metrics[model]
            result[model] = {
                'mae_mean': np.nanmean(metrics['mae']),
                'mae_var': np.nanvar(metrics['mae']),
                'r2_mean': np.nanmean(metrics['r2']),
                'r2_var': np.nanvar(metrics['r2'])
            }
        return result

    # 读取所有数据集数据
    all_data = {ds: read_data(ds) for ds in datasets}

    # ========== 绘图（3行1列） ==========
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))
    fig.suptitle('R² vs MAE for Different Datasets', fontsize=20, fontweight='bold', y=0.98)

    for row_idx, ds in enumerate(datasets):
        ax = axes[row_idx]
        ds_data = all_data[ds]
        r2_min, r2_max = axis_ranges[ds]['r2']
        mae_min = 0

        # 收集当前数据集的有效MAE值，自适应Y轴
        valid_mae = [ds_data[model]['mae_mean'] for model in list_models if not np.isnan(ds_data[model]['mae_mean'])]
        mae_max = max(valid_mae) * 1.1 if valid_mae else 0.01

        # 绘制每个模型的散点
        for idx, model in enumerate(list_models):
            metrics = ds_data[model]
            r2_mean = metrics['r2_mean']
            mae_mean = metrics['mae_mean']
            var_sum = metrics['r2_var'] + metrics['mae_var']

            # 按数据集适配圆大小
            if ds in ['lorenz', 'rossler']:
                circle_size = var_sum * 1e9  # 更窄范围，放大圆大小以区分
            else:
                circle_size = var_sum * 1e5
            circle_size = max(min(circle_size, 8000), 200)  # 限制圆大小范围

            if not np.isnan(r2_mean) and not np.isnan(mae_mean):
                ax.scatter(r2_mean, mae_mean,
                           color=colors[idx], s=circle_size, alpha=0.7,
                           edgecolors='black', linewidth=0.8, label=model if row_idx == 0 else "")

        # ========== 坐标轴精准配置 ==========
        ax.set_xlim(r2_min, r2_max)
        ax.set_ylim(mae_min, mae_max)
        ax.set_xlabel('R² Mean', fontsize=14)
        ax.set_ylabel('MAE Mean', fontsize=14)
        ax.set_title(f'{ds.capitalize()} Dataset (R²: {r2_min} ~ {r2_max})', fontsize=16, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', labelsize=12)

        # 针对窄范围优化刻度（关键）
        if ds in ['lorenz', 'rossler']:
            # 0.9992~1.0005：显示5位小数，间隔0.0002
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.5f}'))
            tick_interval = 0.0002  # 更小的间隔，适配窄范围
            ax.set_xticks(np.arange(r2_min, r2_max + tick_interval/2, tick_interval))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')  # 旋转避免重叠
        else:
            # power：0.92~0.99，显示3位小数，间隔0.02
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
            ax.set_xticks(np.arange(r2_min, r2_max + 0.01, 0.02))

    # ========== 图例 ==========
    handles = [plt.scatter([], [], color=colors[i], s=300, edgecolors='black', linewidth=0.8, label=list_models[i])
               for i in range(len(list_models))]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=12)

    # 布局调整
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.12, right=0.95, hspace=0.4)
    plt.savefig(r'C:\Users\86178\Desktop\Chaotic Net\运行结果\绘图\R2_MAE散点图_终极版.png', dpi=300, bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":
    make_r2_mae_scatter_plot()