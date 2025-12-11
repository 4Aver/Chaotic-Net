import pywt
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset,DataLoader


def get_data(type_name, train_x, train_y, test_x, test_y, time_len,pred_len=1, random_state=42):
    batch_size = 128

    max_len = train_x.shape[0] - pred_len - time_len + 1
    list_x, list_y = [], []
    for i in range(max_len):
        list_x.append(train_x[i:i+time_len,:])
        list_y.append(train_y[i+time_len-1:i+time_len+pred_len-1,:])
    train_x, train_y = np.array(list_x), np.array(list_y).reshape(len(list_y),pred_len)

    max_len = test_x.shape[0] - pred_len - time_len + 1
    list_x, list_y = [], []
    for i in range(max_len):
        list_x.append(test_x[i:i+time_len,:])
        list_y.append(test_y[i+time_len-1:i+time_len+pred_len-1,:])
    test_x, test_y = np.array(list_x), np.array(list_y).reshape(len(list_y),pred_len)


    train_X, valid_X, train_y, valid_y= train_test_split(train_x, train_y, train_size=0.85, random_state=random_state)
    # 将数据转化为Tensor
    # 训练集
    train_seq = torch.from_numpy(np.array(train_X)).type(torch.FloatTensor)
    train_label = torch.from_numpy(np.array(train_y)).type(torch.FloatTensor)
    # 验证集
    valid_seq = torch.from_numpy(np.array(valid_X)).type(torch.FloatTensor)
    valid_label = torch.from_numpy(np.array(valid_y)).type(torch.FloatTensor)
    # 测试集
    test_seq = torch.from_numpy(np.array(test_x)).type(torch.FloatTensor)
    test_label = torch.from_numpy(np.array(test_y)).type(torch.FloatTensor)
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
    np.save(r'C:\Users\86178\Desktop\Chaotic Net\Data\Save_Data\tradition_reconstruction\tr_{}_{}.npy'.format(type_name,random_state),save_dict)


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


def wavelet_decompose_column(column_data, wavelet_name='db4', level=4):
    """
    对单列数据进行小波分解，返回多个分量

    参数:
        column_data (np.array): 单列时间序列数据
        wavelet_name (str): 小波基名称
        level (int): 分解层级

    返回:
        list: 包含多个分量的列表，每个分量都是1D数组
    """
    # 执行小波分解
    coeffs = pywt.wavedec(column_data, wavelet_name, level=level)
    print(len(coeffs))

    # 重构所有分量
    components = []
    for i in range(len(coeffs)):
        # 创建全零系数副本
        coeffs_temp = [np.zeros_like(c) for c in coeffs]

        # 仅保留当前层级系数
        coeffs_temp[i] = coeffs[i].copy()

        # 重构该分量
        component = pywt.waverec(coeffs_temp, wavelet_name)
        components.append(component)

    # 确保所有分量长度一致
    min_len = min(len(column_data), *[len(c) for c in components])
    components = [c[:min_len] for c in components]

    return components


# # 1. 读取数据
# data = pd.read_csv(r'C:\Users\86178\Desktop\Chaotic Net\Data\Original Data\{}.csv'.format('Lorenz'))
# print("原始数据形状:", data.shape)
# print(data.head())
#
# # 2. 准备分解结果容器
# list_all_data = []
#
# # 3. 对每一列进行小波分解
# for i in range(data.shape[1]):
#     column_name = data.columns[i]
#     s = np.array(data[column_name])
#     components = wavelet_decompose_column(s, wavelet_name='sym5', level=5)
#     list_all_data.extend(components)
#     print(f"列 '{column_name}' 分解为 {len(components)} 个分量")
#
#
# all_data = np.array(list_all_data).T
# print("\n所有分量组合后的形状:", all_data.shape)
#
# y = np.array(data.iloc[:, 0]).reshape(-1, 1)  # 目标变量（第一列）
# x = all_data[:-1, :]  # 特征矩阵（从第二行开始）
# y = y[1:, :]  # 目标变量（移除最后一行）
#
# for r in [1,2,3,4,5]:
#     get_data('Wave_lorenz2',x,y,32,3000,1,random_state=r)



# 1. 读取数据
data = pd.read_csv(r'C:\Users\86178\Desktop\Chaotic Net\Data\Original Data\{}.csv'.format('Rossler'))
print("原始数据形状:", data.shape)
print(data.head())

# 2. 准备分解结果容器
list_all_data = []

# 3. 对每一列进行小波分解
for i in range(data.shape[1]):
    column_name = data.columns[i]
    s = np.array(data[column_name])
    components = wavelet_decompose_column(s, wavelet_name='sym5', level=5)
    list_all_data.extend(components)
    print(f"列 '{column_name}' 分解为 {len(components)} 个分量")

all_data = np.array(list_all_data).T
print("\n所有分量组合后的形状:", all_data.shape)

y = np.array(data.iloc[:, 0]).reshape(-1, 1)  # 目标变量（第一列）
x = all_data[:-1, :]  # 特征矩阵（从第二行开始）
y = y[1:, :]  # 目标变量（移除最后一行）

sc = MinMaxScaler(feature_range=(-1, 1))

x = sc.fit_transform(x)
y = sc.fit_transform(y)

train_x, test_x = x[:3000,:], x[3000:,:]
train_y, test_y = y[:3000,:], y[3000:,:]

# 2. 准备分解结果容器
list_all_data_train = []
for i in range(data.shape[1]):
    s = np.array(train_x[i])
    components = wavelet_decompose_column(s, wavelet_name='sym5', level=5)
    list_all_data_train.extend(components)
    print(f"列 '{i}' 分解为 {len(components)} 个分量")
all_data_train = np.array(list_all_data_train).T
print("\n所有分量组合后的形状:", all_data_train.shape)
train_x = all_data_train[:, :]  # 特征矩阵（从第二行开始）


list_all_data_test = []
for i in range(data.shape[1]):
    s = np.array(test_x[i])
    components = wavelet_decompose_column(s, wavelet_name='sym5', level=5)
    list_all_data_test.extend(components)
    print(f"列 '{i}' 分解为 {len(components)} 个分量")
all_data_test = np.array(list_all_data_test).T
print("\n所有分量组合后的形状:", all_data_test.shape)
test_x = all_data_test[:, :]  # 特征矩阵（从第二行开始）


for r in [1,2,3,4,5]:
    get_data('Wave_rossler2',train_x, train_y, test_x, test_y,32,1,random_state=r)




# # 1. 读取数据
# data = pd.read_csv(r'C:\Users\86178\Desktop\Chaotic Net\Data\Original Data\{}.csv'.format('Power'))
# print("原始数据形状:", data.shape)
# print(data.head())
# y = np.array(data.iloc[1:, 3]).reshape(-1, 1)  # 目标变量（第一列）
#
# sc = MinMaxScaler(feature_range=(-1, 1))
# data = data.iloc[:-1,:]
#
# data = sc.fit_transform(data)
# y = sc.fit_transform(y)
#
# train_x, test_x = data[:3000,:].T, data[3000:,:].T
# train_y, test_y = y[:3000,:], y[3000:,:]
#
# # 2. 准备分解结果容器
# list_all_data_train = []
# for i in range(data.shape[1]):
#     s = np.array(train_x[i])
#     components = wavelet_decompose_column(s, wavelet_name='sym5', level=5)
#     list_all_data_train.extend(components)
#     print(f"列 '{i}' 分解为 {len(components)} 个分量")
# all_data_train = np.array(list_all_data_train).T
# print("\n所有分量组合后的形状:", all_data_train.shape)
# train_x = all_data_train[:, :]  # 特征矩阵（从第二行开始）
#
#
# list_all_data_test = []
# for i in range(data.shape[1]):
#     s = np.array(test_x[i])
#     components = wavelet_decompose_column(s, wavelet_name='sym5', level=5)
#     list_all_data_test.extend(components)
#     print(f"列 '{i}' 分解为 {len(components)} 个分量")
# all_data_test = np.array(list_all_data_test).T
# print("\n所有分量组合后的形状:", all_data_test.shape)
# test_x = all_data_test[:, :]  # 特征矩阵（从第二行开始）
#
#
# for r in [1,2,3,4,5]:
#     get_data('Wave_power2',train_x, train_y, test_x, test_y,32,1,random_state=r)
