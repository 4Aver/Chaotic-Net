import time
import torch
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from 对比实验.utils import get_all_result

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 环境设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 使用gpu
batch_size = 128
criterion = nn.MSELoss()
torch.manual_seed(21)       # 为CPU中设置种子，生成随机数。一旦固定种子，后面依次生成的随机数其实都是固定的。后面在生成loader时打乱顺序每次运行的结果一样
np.random.seed(21)

start_time = time.time()


def train(model, optimizer, scheduler, epoch, train_dataloader, train_seq, type_loss):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    for batch_index,batch_data in enumerate(train_dataloader):
        data,targets = batch_data       # torch.Size([B, L, D]) torch.Size([B, pred_len])

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # 防止梯度爆炸或梯度消失问题
        optimizer.step()
        total_loss += loss.item()
        log_interval = int(len(train_seq) / batch_size/5 )
        if batch_index % log_interval == 0 and batch_index > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f}'.format(
                epoch, batch_index, len(train_seq) // batch_size, scheduler.get_lr()[0],
                              elapsed * 1000 / log_interval,cur_loss))  # , math.exp(cur_loss)
            total_loss = 0
            start_time = time.time()
    return model


def evaluate(eval_model, dataloader, data_source_x,type_loss):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for batch_index,dataset in enumerate(dataloader):
            data, targets = dataset

            output = eval_model(data)

            loss = criterion(output, targets)
            total_loss +=  len(data)*loss.cpu().item()

            result = torch.cat((result, output.cpu()),0)  # todo: check this. -> looks good to me
            truth = torch.cat((truth, targets.cpu()), 0)
    return total_loss / len(data_source_x),result,truth


def get_model(type_name, model_name, load_name):
    model = None        # 定义模型

    if load_name == 'sr':
        if model_name == 'Chaotic-Net':
            from 研究生课题.前沿研究.Chaotic_Net.models.independent_Transformer import model_1
            if type_name == 'lorenz':
                model = model_1(time_len=32, M=3, list_input_dims=[5, 5, 5])
            if type_name == 'power':
                model = model_1(time_len=32, M=6, list_input_dims=[5, 5, 5, 5, 5, 5])

        if model_name == 'LSTM':
            from 研究生课题.前沿研究.Chaotic_Net.对比实验.test1_LSTM import LSTM_Model
            if type_name == 'lorenz':
                model = LSTM_Model(seq_len=15, input_dim=15, hidden_dim1=128, hidden_dim2=16, pred_len=1)
            if type_name == 'power':
                model = LSTM_Model(seq_len=32, input_dim=30, hidden_dim1=128, hidden_dim2=16, pred_len=1)

        if model_name == 'Transformer':
            from 研究生课题.前沿研究.Chaotic_Net.对比实验.test2_Transformer import Transformer_Model
            if type_name == 'lorenz':
                model = Transformer_Model(seq_len=32, pred_len=1, enc_in=15)
            if type_name == 'power':
                model = Transformer_Model(seq_len=32, pred_len=1, enc_in=30)

    else:
        if model_name == 'LSTM':
            from 研究生课题.前沿研究.Chaotic_Net.对比实验.test1_LSTM import LSTM_Model
            if type_name == 'lorenz':
                model = LSTM_Model(seq_len=15, input_dim=15, hidden_dim1=128, hidden_dim2=16, pred_len=1)
            if type_name == 'power':
                model = LSTM_Model(seq_len=32, input_dim=30, hidden_dim1=128, hidden_dim2=16, pred_len=1)

        if model_name == 'Transformer':
            from 研究生课题.前沿研究.Chaotic_Net.对比实验.test2_Transformer import Transformer_Model
            if type_name == 'lorenz':
                model = Transformer_Model(seq_len=32, pred_len=1, enc_in=15)
            if type_name == 'power':
                model = Transformer_Model(seq_len=32, pred_len=1, enc_in=30)

    return model


def read_npy(type_name,model_name,epochs,load_name,weight_decay,lr = 0.001,seed=0,pred_len=1,time_len=32):
    # saved_dict = np.load(r'C:\Users\86178\Desktop\Chaotic Net\Data\Save_Data\tradition_reconstruction\tr_{}_32(val).npy'.format(type_name),allow_pickle=True).item()
    saved_dict = np.load(r'C:\Users\86178\Desktop\Chaotic Net\Data\Save_Data\segment_reconstruction\sr_{}_32(val).npy'.format(type_name),allow_pickle=True).item()

    train_seq = saved_dict['train']['train_seq']
    train_label = saved_dict['train']['train_label']
    train_dataloader = saved_dict['train']['train_dataloader']

    val_seq = saved_dict['val']['val_seq']
    val_label = saved_dict['val']['val_label']
    val_dataloader = saved_dict['val']['val_dataloader']

    test_seq = saved_dict['test']['test_seq']
    test_label = saved_dict['test']['test_label']
    test_dataloader = saved_dict['test']['test_dataloader']

    print('----------train--------')
    print(f'train X shape:{train_seq.shape}')
    print(f'train y shape:{train_label.shape}')

    print('----------val--------')
    print(f'val X shape:{val_seq.shape}')
    print(f'val y shape:{val_label.shape}')

    print('----------test--------')
    print(f'test X shape:{test_seq.shape}')
    print(f'test y shape:{test_label.shape}')




    type_loss = 'mse'
    # type_loss = 'dilate'
    model = get_model(type_name, model_name, load_name,pred_len=pred_len,time_len=time_len).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.98)       # 学习率调度器对象,调度器会在每个10个epoch后将学习率乘以gamma。
    best_test_loss = 1000
    best_val_output, best_val_target,best_test_output, best_test_target = 0, 0, 0, 0
    best_model = None
    list_train_loss, list_test_loss = [], []

    # 训练模型
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_model = train(model, optimizer, scheduler, epoch,train_dataloader, train_seq,type_loss)

        train_loss, train_output, train_target = evaluate(train_model, train_dataloader, train_seq,type_loss)
        test_loss, test_output, test_target = evaluate(train_model, test_dataloader, test_seq,type_loss)
        list_train_loss.append(train_loss)
        list_test_loss.append(test_loss)

        if test_loss<best_test_loss:
            best_test_loss = test_loss
            best_test_output = test_output
            best_test_target = test_target
            best_model = train_model

        print(
            '| end of epoch {:3d} | time: {:5.2f}s  | train loss {:.10f}| test loss {:.10f} '.format(
                epoch, (time.time() - epoch_start_time), train_loss, test_loss))
        print('-' * 89)

        scheduler.step()        # 学习率调度器对象

    mse, rmse, mae, mape, r2 = get_all_result(best_test_target, best_test_output)
    data_out = pd.DataFrame({'true':[i[0] for i in best_test_target.tolist()],
                             'pred':[i[0] for i in best_test_output.tolist()]},
                            index=range(len(best_test_output)))
    print(data_out)

    plt.plot(best_test_target, label='真实值')
    plt.plot(best_test_output, label='预测值')
    plt.legend()
    plt.show()

