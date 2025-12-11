import torch
import numpy as np
import pandas as pd
from PyEMD.EMD import EMD
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import train_test_split


def calculate_IMFs(s):
    # 配置EMD参数
    emd = EMD()
    emd.MAXIMUM_ITERATIONS = 1000  # 增加最大迭代次数
    emd.RANGE_THR = 0.05  # 调整停止条件参数

    # 执行EMD分解
    imfs = emd(s, max_imf=10)  # 尝试提取最多10个IMF

    # 检查实际得到的IMF数量
    n_imfs = imfs.shape[0]
    if n_imfs < 10:
        print(f"警告：实际只分解出{n_imfs}个IMF")
        print("尝试调整EMD参数或检查数据特性")

    # 计算各IMF与原始序列的相关性
    correlations = []
    for i, imf in enumerate(imfs):
        corr, _ = pearsonr(imf, s)
        correlations.append(round(corr, 4))

    # 输出相关性结果
    print("\nIMF与原始序列的相关系数：")
    for i, corr in enumerate(correlations):
        label = f"IMF {i + 1}" if i < n_imfs - 1 else "Residual"
        print(f"{label}: {corr}")

    def make_polt():
        # 可视化设置
        rows = 5  # 调整行数以适应显示
        cols = 2
        fig, axs = plt.subplots(rows, cols, figsize=(16, 20))
        fig.suptitle('IMFs with Correlation Coefficients', y=1.02, fontsize=14)

        # 绘制各IMF和残差
        for i in range(rows * cols):
            row = i // cols
            col = i % cols
            if i < len(imfs):
                axs[row, col].plot(imfs[i], linewidth=0.8)
                title = f'IMF {i + 1} (ρ={correlations[i]})' if i < n_imfs - 1 else f'Residual (ρ={correlations[i]})'
                axs[row, col].set_title(title, fontsize=10)
                axs[row, col].tick_params(axis='both', which='major', labelsize=8)
            else:
                axs[row, col].axis('off')

        plt.tight_layout()
        plt.savefig('EMD_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    # make_polt()

    return imfs
# imfs = calculate_IMFs()


def reconstruct_signal(imfs, selected_indices, include_residual=False):
    """
    重构选定IMF组合的新序列

    参数：
    imfs -- 从EMD分解得到的IMF矩阵（二维数组）
    selected_indices -- 需要选择的IMF索引列表（从0开始计数）
    include_residual -- 是否包含残差项（默认False）

    返回：
    reconstructed -- 重构后的新序列
    """
    # 输入校验
    if not isinstance(selected_indices, (list, tuple, np.ndarray)):
        raise TypeError("selected_indices必须是列表、元组或数组")

    max_idx = imfs.shape[0] - 1 if include_residual else imfs.shape[0] - 2
    invalid_indices = [i for i in selected_indices if i > max_idx]
    if invalid_indices:
        raise ValueError(f"索引{invalid_indices}超出有效范围(0-{max_idx})")

    # 选择指定IMF
    selected_imfs = imfs[selected_indices]

    # 包含残差处理
    if include_residual:
        residual = imfs[-1]
        return np.sum(selected_imfs, axis=0) + residual

    return np.sum(selected_imfs, axis=0)



def reconstruct_data(imfs,selected_indices,s):
    new_series = reconstruct_signal(imfs, selected_indices)

    def make_plot():
        # 可视化对比
        plt.figure(figsize=(12, 6))
        plt.plot(s, label='Original', alpha=0.6, linewidth=1)
        plt.plot(new_series, label='Reconstructed (IMF 1-3)',
                 linestyle='--', linewidth=1.2)
        plt.title(f'Signal Reconstruction (Using IMF {[x + 1 for x in selected_indices]})')
        plt.legend()
        plt.tight_layout()
        plt.savefig('reconstruction_comparison.png', dpi=300)
        plt.show()
    make_plot()

    # 高级用法：计算重构信号占比
    original_energy = np.sum(s ** 2)
    reconstructed_energy = np.sum(new_series ** 2)
    energy_ratio = reconstructed_energy / original_energy

    print(f"重构信号能量占比：{energy_ratio:.2%}")
    print(f"选用的IMF数量：{len(selected_indices)}/{imfs.shape[0]}")
    print(f"相关系数：{pearsonr(new_series, s)[0]:.4f}")

    return new_series
    # 保存重构结果
    # np.save('reconstructed_signal.npy', new_series)


def get_data(type_name,x, y, time_len,data_len,pred_len=1,seed=42):
    batch_size = 128
    sc = MinMaxScaler(feature_range=(-1, 1))
    x = sc.fit_transform(x)
    y = sc.fit_transform(y)

    max_len = x.shape[0] - pred_len - time_len + 1
    list_x, list_y = [], []
    for i in range(max_len):
        list_x.append(x[i:i+time_len,:])
        list_y.append(y[i+time_len-1:i+time_len+pred_len-1,:])
    x, y = np.array(list_x), np.array(list_y).reshape(len(list_y),pred_len)

    train_data = x[:data_len, :]
    test_data = x[data_len:, :]

    train_label = y[:data_len, :]
    test_label = y[data_len:, :]

    train_X, valid_X, train_y, valid_y= train_test_split(train_data, train_label, train_size=0.85, random_state=seed)
    # 将数据转化为Tensor
    # 训练集
    train_seq = torch.from_numpy(np.array(train_X)).type(torch.FloatTensor)
    train_label = torch.from_numpy(np.array(train_y)).type(torch.FloatTensor)
    # 验证集
    valid_seq = torch.from_numpy(np.array(valid_X)).type(torch.FloatTensor)
    valid_label = torch.from_numpy(np.array(valid_y)).type(torch.FloatTensor)
    # 测试集
    test_seq = torch.from_numpy(np.array(test_data)).type(torch.FloatTensor)
    test_label = torch.from_numpy(np.array(test_label)).type(torch.FloatTensor)
    print('----------train--------')
    print(f'train X shape:{train_seq.shape}')
    print(f'train y shape:{train_label.shape}')

    print('----------val--------')
    print(f'val X shape:{valid_seq.shape}')
    print(f'val y shape:{valid_label.shape}')

    print('----------test--------')
    print(f'test X shape:{test_seq.shape}')
    print(f'test y shape:{test_label.shape}')

    train_dataset = TensorDataset(train_seq,train_label)
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

    val_dataset = TensorDataset(valid_seq,valid_label)
    val_dataloader = DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)

    test_dataset = TensorDataset(test_seq,test_label)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)
    save_dict = {'train':{'train_seq':train_seq,'train_label':train_label,'train_dataloader':train_dataloader},
                 'val': {'val_seq': valid_seq, 'val_label': valid_label, 'val_dataloader': val_dataloader},
                 'test':{'test_seq':test_seq,'test_label':test_label,'test_dataloader':test_dataloader}
                 }
    np.save(r'C:\Users\86178\Desktop\Chaotic Net\Data\Save_Data\tradition_reconstruction\tr_{}_{}.npy'.format(type_name,seed),save_dict)


    # # 将数据转化为Tensor
    # # 训练集
    # train_seq = torch.from_numpy(np.array(train_data)).type(torch.FloatTensor)
    # train_label = torch.from_numpy(np.array(train_label)).type(torch.FloatTensor)
    #
    # # 测试集
    # test_seq = torch.from_numpy(np.array(test_data)).type(torch.FloatTensor)
    # test_label = torch.from_numpy(np.array(test_label)).type(torch.FloatTensor)
    # print('----------train--------')
    # print(f'train X shape:{train_seq.shape}')
    # print(f'train y shape:{train_label.shape}')
    #
    # print('----------test--------')
    # print(f'test X shape:{test_seq.shape}')
    # print(f'test y shape:{test_label.shape}')
    #
    # train_dataset = TensorDataset(train_seq,train_label)
    # train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    #
    # test_dataset = TensorDataset(test_seq,test_label)
    # test_dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)
    #
    # save_dict = {'train':{'train_seq':train_seq,'train_label':train_label,'train_dataloader':train_dataloader},
    #              'test':{'test_seq':test_seq,'test_label':test_label,'test_dataloader':test_dataloader}
    #              }
    # np.save(r'C:\Users\86178\Desktop\Chaotic Mamba\Data\Save_Data\tradition_reconstruction\tr_{}_32.npy'.format(type_name),save_dict)


data = pd.read_csv(r'C:\Users\86178\Desktop\Chaotic Net\Data\Original Data\{}.csv'.format('Lorenz'))
print(data)

list_all_data = []
y = np.array(data.iloc[:,0]).reshape(-1,1)
for i in range(3):
    s = np.array(data.iloc[:,i]).reshape(-1,)
    imfs = np.array(calculate_IMFs(s))
    list_all_data.extend(imfs)

all_data = np.array(list_all_data).T
# print(all_data)

x = all_data[:-1,:]
y = y[1:,:]
print(x.shape,y.shape)

get_data('EMD_lorenz2',x,y,32,3000,1)

for i in [1,2,3,4,5]:
    get_data('EMD_lorenz2', x, y, 32, 3000, 1, seed=i)


# data = pd.read_csv(r'C:\Users\86178\Desktop\Chaotic Net\Data\Original Data\{}.csv'.format('Rossler'))
# print(data)
#
# list_all_data = []
# y = np.array(data.iloc[:,0]).reshape(-1,1)
# for i in range(3):
#     s = np.array(data.iloc[:,i]).reshape(-1,)
#     imfs = np.array(calculate_IMFs(s))
#     list_all_data.extend(imfs)
#
# all_data = np.array(list_all_data).T
# # print(all_data)
#
# x = all_data[:-1,:]
# y = y[1:,:]
# print(x.shape,y.shape)
#
# get_data('EMD_rossler2',x,y,32,3000,1)
#
# for i in [1,2,3,4,5]:
#     get_data('EMD_rossler2', x, y, 32, 3000, 1, seed=i)


# data = pd.read_csv(r'C:\Users\86178\Desktop\Chaotic Net\Data\Original Data\{}.csv'.format('Power'))
# print(data)
#
# list_all_data = []
# # list_all_data.append(np.array(data.iloc[:,3]).reshape(-1,))     # 方便后续切分预测数据
#
# y = np.array(data.iloc[:,3]).reshape(-1,1)
#
# plt.plot(y)
# plt.show()
#
#
# for i in [0,1,2,3,4,5]:
#     print(i)
#     s = np.array(data.iloc[:,i]).reshape(-1,)
#     if i == 2:      # 不需要进行EMD分解
#         list_all_data.append(s)
#         continue
#
#     imfs = np.array(calculate_IMFs(s))
#     list_all_data.extend(imfs)
#     print('\n')
#
# all_data = np.array(list_all_data).T
# print(all_data.shape)
#
# x = all_data[1:,:]
# y = y[:-1,:]
# print(x.shape,y.shape)
#
#
# for i in [1,2,3,4,5]:
#     get_data('EMD_power2',x,y,32,3000,1,seed=i)

'''
lorenz [5,6,6]
rossler [5,5,6]
power [9,9,1,9,10,10]
'''
