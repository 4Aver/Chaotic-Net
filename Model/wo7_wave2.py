import pywt
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


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
    # print(len(coeffs))

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


def get_data(type_name, train_x, train_y, test_x, test_y, time_len, pred_len=1, random_state=42, wavelet_level=3):
    """
    修改后的数据获取函数，在时间窗口内进行小波分解

    参数:
        wavelet_level (int): 小波分解层级（根据时间窗口调整）
    """
    batch_size = 128

    # 创建时间窗口
    def create_windows(data, labels):
        max_len = data.shape[0] - pred_len - time_len + 1
        list_x, list_y = [], []
        for i in range(max_len):
            window = data[i:i + time_len, :]

            components = wavelet_decompose_column(window.T, wavelet_name='sym5', level=5)
            # print(len(components),len(components[0]),len(components[0][0]))
            x = np.array(components).reshape(-1,time_len).T
            # print(x.shape)

            # 将分解结果转换为特征矩阵 [time_len, n_components*n_features]
            # window_features = np.hstack(components)
            list_x.append(x)
            list_y.append(labels[i + time_len - 1:i + time_len + pred_len - 1, :])

        return np.array(list_x), np.array(list_y).reshape(len(list_y), pred_len)

    # 处理训练集和测试集
    train_x, train_y = create_windows(train_x, train_y)
    test_x, test_y = create_windows(test_x, test_y)

    # 分割训练集和验证集
    train_X, valid_X, train_y, valid_y = train_test_split(
        train_x, train_y, train_size=0.85, random_state=random_state
    )

    # 将数据转化为Tensor
    train_seq = torch.from_numpy(train_X).type(torch.FloatTensor)
    train_label = torch.from_numpy(train_y).type(torch.FloatTensor)
    valid_seq = torch.from_numpy(valid_X).type(torch.FloatTensor)
    valid_label = torch.from_numpy(valid_y).type(torch.FloatTensor)
    test_seq = torch.from_numpy(test_x).type(torch.FloatTensor)
    test_label = torch.from_numpy(test_y).type(torch.FloatTensor)

    print('----------train--------')
    print(f'train X shape:{train_seq.shape}')
    print(f'train y shape:{train_label.shape}')
    print('----------val--------')
    print(f'val X shape:{valid_seq.shape}')
    print(f'val y shape:{valid_label.shape}')
    print('----------test--------')
    print(f'test X shape:{test_seq.shape}')
    print(f'test y shape:{test_label.shape}')

    # 创建DataLoader
    train_dataset = TensorDataset(train_seq, train_label)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(valid_seq, valid_label)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = TensorDataset(test_seq, test_label)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 保存数据
    save_dict = {
        'train': {'train_seq': train_seq, 'train_label': train_label, 'train_dataloader': train_dataloader},
        'val': {'val_seq': valid_seq, 'val_label': valid_label, 'val_dataloader': val_dataloader},
        'test': {'test_seq': test_seq, 'test_label': test_label, 'test_dataloader': test_dataloader}
    }
    # np.save(
    #     f'C:/Users/86178/Desktop/Chaotic Net/Data/Save_Data/tradition_reconstruction/tr_{type_name}_{random_state}_({pred_len}).npy',
    #     save_dict)
    np.save(
        f'C:/Users/86178/Desktop/Chaotic Net/Data/Save_Data/tradition_reconstruction/tr_{type_name}_({time_len}_{pred_len}).npy',
        save_dict)


def wavelet_decompose_window(window_data, wavelet_name='sym5', level=3):
    """
    对时间窗口内的数据进行小波分解（短序列优化版）

    参数:
        window_data (np.array): 时间窗口内的单列数据 (长度=time_len)
        wavelet_name (str): 小波基名称
        level (int): 分解层级

    返回:
        np.array: 分解后的分量矩阵 [time_len, n_components]
    """
    # 调整层级以适应短窗口
    max_level = pywt.dwt_max_level(len(window_data), pywt.Wavelet(wavelet_name).dec_len)
    level = min(level, max_level)

    if level == 0:
        # 无法分解，直接返回原始数据
        return window_data.reshape(-1, 1)

    # 执行小波分解
    coeffs = pywt.wavedec(window_data, wavelet_name, level=level, mode='periodization')

    # 重构所有分量
    components = []
    for i in range(len(coeffs)):
        # 创建全零系数副本
        coeffs_temp = [np.zeros_like(c) for c in coeffs]
        coeffs_temp[i] = coeffs[i].copy()

        # 重构该分量
        component = pywt.waverec(coeffs_temp, wavelet_name, mode='periodization')

        # 确保长度与窗口一致
        component = component[:len(window_data)]
        components.append(component)

    return np.column_stack(components)


# 主数据处理流程
def main_processing(dataset_name='Power', time_len=32, pred_len=1, wavelet_level=3):
    # 1. 读取数据
    data = pd.read_csv(f'C:/Users/86178/Desktop/Chaotic Net/Data/Original Data/{dataset_name}.csv')
    print(f"原始数据形状: {data.shape}")

    # 2. 准备目标变量Y（保持不变）
    if dataset_name == 'Lorenz':
        # Lorenz数据：第一列作为目标
        y = data.iloc[:, 0].values.reshape(-1, 1)
        # 移除最后一行以对齐
        data = data.iloc[:-1, :]

    if dataset_name == 'Rossler':
        # Lorenz数据：第一列作为目标
        y = data.iloc[:, 0].values.reshape(-1, 1)
        # 移除最后一行以对齐
        data = data.iloc[:-1, :]

    elif dataset_name == 'Power':
        # Power数据：第四列作为目标
        y = data.iloc[1:, 3].values.reshape(-1, 1)
        # 移除最后一行以对齐
        data = data.iloc[:-1, :]

    # 3. 归一化处理
    sc_x = MinMaxScaler(feature_range=(-1, 1))
    sc_y = MinMaxScaler(feature_range=(-1, 1))

    data_scaled = sc_x.fit_transform(data)
    y_scaled = sc_y.fit_transform(y)

    # 4. 分割训练集和测试集
    train_size = 3000
    train_x, test_x = data_scaled[:train_size, :], data_scaled[train_size:, :]
    train_y, test_y = y_scaled[:train_size], y_scaled[train_size:]

    print(f"训练集X形状: {train_x.shape}, Y形状: {train_y.shape}")
    print(f"测试集X形状: {test_x.shape}, Y形状: {test_y.shape}")

    # # 5. 对每个随机种子运行
    # for r in [2, 3, 4, 5,6,7,8,9,10,11,12]:
    #     print(f"\n处理随机种子 {r}...")
    #     get_data(
    #         f'Wave_{dataset_name.lower()}',
    #         train_x, train_y,
    #         test_x, test_y,
    #         time_len=time_len,
    #         pred_len=r,
    #         random_state=42,
    #         wavelet_level=wavelet_level
    #     )
    get_data(
        f'Wave_{dataset_name.lower()}',
        train_x, train_y,
        test_x, test_y,
        time_len=time_len,
        pred_len=6,
        random_state=42,
        wavelet_level=wavelet_level
    )


# 执行处理
if __name__ == "__main__":
    # 处理Lorenz数据集
    # list_time = [24, 32, 40, 48, 56, 64]
    list_time = [8,16]

    for time_len in list_time:
        print("处理Lorenz数据集...")
        main_processing(dataset_name='Lorenz', time_len=time_len, wavelet_level=4)

        # # 处理Power数据集
        print("\n处理Power数据集...")
        main_processing(dataset_name='Power', time_len=time_len, wavelet_level=4)

        # 处理Lorenz数据集
        # print("处理Rossler数据集...")
        # main_processing(dataset_name='Rossler', time_len=time_len, wavelet_level=4)

    # print("处理Lorenz数据集...")
    # main_processing(dataset_name='Lorenz', time_len=32, wavelet_level=4)
    #
    # # # # 处理Power数据集
    # # print("\n处理Power数据集...")
    # # main_processing(dataset_name='Power', time_len=32, wavelet_level=4)
    #
    # # # 处理Lorenz数据集
    # # print("处理Rossler数据集...")
    # # main_processing(dataset_name='Rossler', time_len=32, wavelet_level=4)
