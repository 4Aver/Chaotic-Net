import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# --------------------------
# 1. 基础设置（解决字体和显示问题）
# --------------------------
plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示
plt.rcParams['font.size'] = 12  # 全局字体大小
# plt.rcParams['figure.dpi'] = 100  # 提高图像清晰度，避免"空图"视觉效果
warnings.filterwarnings('ignore')


# --------------------------
# 2. 核心函数（确保数据正确计算和读取）
# --------------------------
def get_all_result(true, pred):
    """计算RMSE和MAE（保留核心指标，移除冗余）"""
    mse = np.mean((true - pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(true - pred))
    return rmse, mae


def get_max_min(type_name):
    """获取反归一化所需的最大/最小值（严格复用原逻辑）"""
    data_path = r'C:\Users\86178\Desktop\Chaotic Net\Data\Original Data\{}.csv'.format(type_name)
    # 增加路径检查提示（避免因路径错误导致数据读取失败）
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"警告：原始数据文件未找到，请检查路径：{data_path}")
        return 1, 0  # 临时返回默认值，避免程序崩溃

    # 按原逻辑选择目标列（power取第4列，其他取第1列）
    if type_name == 'power':
        target_col = data.iloc[:, 3]
    else:
        target_col = data.iloc[:, 0]
    return target_col.max(), target_col.min()


def read_metrics(type_name, num, k):
    """读取CSV并计算指标（增加异常捕获，确保数据有效）"""
    data_len = 800
    # 拼接CSV文件路径（严格匹配你的文件命名规则）
    csv_path = r'C:\Users\86178\Desktop\Chaotic Net\运行结果\预测结果\超参实验\sr_{}_Chaotic-Net_MOEE_{}_{}.csv'.format(
        type_name, num, k)

    # 捕获文件读取异常（避免因文件缺失导致矩阵全为NaN）
    try:
        data = pd.read_csv(csv_path)
        # 截取最后800行的第2-3列（true和pred列，对应原逻辑iloc[-data_len:, 1:]）
        if len(data) < data_len:
            print(f"警告：文件{csv_path}数据不足{data_len}行，使用全部数据")
            true = data.iloc[:, 1].values  # true列
            pred = data.iloc[:, 2].values  # pred列
        else:
            true = data.iloc[-data_len:, 1].values
            pred = data.iloc[-data_len:, 2].values
    except FileNotFoundError:
        print(f"错误：CSV文件未找到，请检查路径：{csv_path}")
        return np.nan, np.nan  # 缺失文件返回NaN

    # 反归一化（严格复用原逻辑）
    data_max, data_min = get_max_min(type_name)
    true = (true + 1) * (data_max - data_min) / 2 + data_min
    pred = (pred + 1) * (data_max - data_min) / 2 + data_min

    # 计算并返回指标
    return get_all_result(true, pred)


# --------------------------
# 3. 批量读取数据并构建矩阵（确保数据填充正确）
# --------------------------
# 定义数据类型列表
data_types = ['lorenz', 'power']
data_labels = ['Lorenz', 'Power']

# 定义参数范围
num_list = range(1, 9)  # X轴：num_experts（1-8）
k_list = range(1, 9)  # Y轴：k（1-8）

# 初始化存储矩阵的字典
rmse_mats = {}
mae_mats = {}

# 为每种数据类型创建矩阵
for type_name in data_types:
    # 初始化8x8矩阵（存储RMSE和MAE，初始为NaN）
    rmse_mat = np.full((len(k_list), len(num_list)), np.nan)
    mae_mat = np.full((len(k_list), len(num_list)), np.nan)

    # 填充矩阵（只处理num ≥ k的下三角区域）
    for num in num_list:
        for k in k_list:
            if num >= k:  # 仅保留有效组合（num_experts ≥ k）
                # 转换为矩阵索引（num从1→0，k从1→0）
                num_idx = num - 1  # X轴索引（对应列）
                k_idx = k - 1  # Y轴索引（对应行）
                # 读取指标并填入矩阵
                rmse, mae = read_metrics(type_name, num, k)
                rmse_mat[k_idx, num_idx] = rmse
                mae_mat[k_idx, num_idx] = mae

    # 存储矩阵
    rmse_mats[type_name] = rmse_mat
    mae_mats[type_name] = mae_mat

    # 检查矩阵数据（避免"空图"：打印非NaN值数量，确认数据已填充）
    print(f"{type_name.upper()} - RMSE矩阵有效数据量：{np.sum(~np.isnan(rmse_mat))} 个")
    print(f"{type_name.upper()} - MAE矩阵有效数据量：{np.sum(~np.isnan(mae_mat))} 个")

# --------------------------
# 4. 绘制热力图（2行2列布局）
# --------------------------
# 创建2行2列子图（宽度足够，避免数据拥挤）
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Evaluation Metrics Comparison: Lorenz vs Power', fontsize=16, y=0.98)

mask = ~np.triu(np.ones_like(rmse_mats['lorenz'], dtype=bool))

# 为每种数据类型绘制热力图
for i, type_name in enumerate(data_types):
    # 获取当前数据类型的矩阵
    rmse_mat = rmse_mats[type_name]
    mae_mat = mae_mats[type_name]
    if type_name == 'lorenz':
        fmt = ".4f"
    else:
        fmt = ".1f"

    # --------------------------
    # 左侧：RMSE热力图
    # --------------------------
    sns.heatmap(
        rmse_mat,
        ax=axes[i, 0],
        mask=mask,  # 应用下三角掩码
        annot=True,  # 显示具体数值
        fmt=fmt,  # 数值保留4位小数
        cmap="YlGnBu",
        xticklabels=num_list,  # X轴标签：num_experts（1-8）
        yticklabels=k_list,  # Y轴标签：k（1-8）
        cbar_kws={"label": "RMSE", "shrink": 0.8}  # 颜色条设置
    )
    # 关键：反转Y轴，让k=1在顶部，k=8在底部（从小到大排列）
    axes[i, 0].invert_yaxis()
    # 设置标签和标题
    axes[i, 0].set_xlabel('num_experts', fontsize=12, fontweight='bold')
    axes[i, 0].set_ylabel('k', fontsize=12, fontweight='bold')
    axes[i, 0].set_title(f'{data_labels[i]} - RMSE Heatmap', fontsize=14, fontweight='bold')

    # --------------------------
    # 右侧：MAE热力图
    # --------------------------
    sns.heatmap(
        mae_mat,
        ax=axes[i, 1],
        mask=mask,
        annot=True,
        fmt=fmt,
        cmap="YlOrRd",  # MAE用不同色系（橙红色系），与RMSE区分
        xticklabels=num_list,
        yticklabels=k_list,
        cbar_kws={"label": "MAE", "shrink": 0.8},
    )
    axes[i, 1].invert_yaxis()  # 同样反转Y轴
    axes[i, 1].set_xlabel('num_experts', fontsize=15, fontweight='bold')
    axes[i, 1].set_ylabel('k', fontsize=15, fontweight='bold')
    axes[i, 1].set_title(f'{data_labels[i]} - MAE Heatmap', fontsize=14, fontweight='bold')

# --------------------------
# 调整布局（避免标签被截断）
# --------------------------
plt.tight_layout()
plt.savefig(r'C:\Users\86178\Desktop\Chaotic Net\运行结果\绘图\实验结果图\超参数实验2.png', dpi=300)
# plt.show()