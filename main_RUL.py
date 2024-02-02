import os
import sys
import platform
current_system = platform.system()
if current_system == "Linux":

    current_directory = os.getcwd()
    new_directory = "/enc/y_song/work/pytorch_geometric_temporal/"
    os.chdir(new_directory)
    print("current_directory:", new_directory)



import numpy as np
import random as rn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']



# train数据读取#######################################
train_df = pd.read_csv('train_FD001.txt', sep=" ", header=None)
train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                    's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                    's15', 's16', 's17', 's18', 's19', 's20', 's21']
train_df = train_df.sort_values(['id', 'cycle'])
# train数据读取#######################################

# test 数据读取#######################################
test_df = pd.read_csv('test_FD001.txt', sep=" ", header=None)
test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                   's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                   's15', 's16', 's17', 's18', 's19', 's20', 's21']
# test 数据读取#######################################

# lable数据读取#######################################
truth_df = pd.read_csv('RUL_FD001.txt', sep=" ", header=None)
truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)
# lable数据读取#######################################

# train数据处理#######################################
rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
# 获得各个id下cycle的最大值，shape（100，2）
rul.columns = ['id', 'max']
# 将rul的列抬头由cycle换为max
train_df = train_df.merge(rul, on=['id'], how='left')
# 根据id这列，将max这列放到train_df的末尾，对于每个id不同cycle，max的值是一直的
train_df['RUL'] = train_df['max'] - train_df['cycle']
# 新增一列RUL，用当前的max减去当前的cycle
train_df.drop('max', axis=1, inplace=True)
train_df['cycle_norm'] = train_df['cycle']
# 新增cycle_norm这列,作为自变量之一
cols_normalize = train_df.columns.difference(['id', 'cycle', 'RUL'])
# 找出需要标准化的列
min_max_scaler = preprocessing.MinMaxScaler((0, 1))
norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]),
                             columns=cols_normalize,
                             index=train_df.index)
# 对需要标准化的列进行标准化
join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
# 将没标准化的列和标准化的列合并
train_df = join_df.reindex(columns=train_df.columns)
# 根据train_df.columns的顺序对各列进行重新排序

# 将RUL中大于130的值改为130，cycle_norm和RUL无关，所以不管
train_df['RUL'].loc[train_df['RUL'] > 125] = 125
# train数据处理#######################################

# test 数据处理#######################################
test_df['cycle_norm'] = test_df['cycle']
norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]),
                            columns=cols_normalize,
                            index=test_df.index)
test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
test_df = test_join_df.reindex(columns=test_df.columns)
test_df = test_df.reset_index(drop=True)
rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
truth_df.columns = ['more']
# 将truth_df的RUL一列抬头改为more
truth_df['id'] = truth_df.index + 1
# 原truth_df的索引为0~99，新增一个id列，其值为1~100
truth_df['max'] = rul['max'] + truth_df['more']
# 如第一个零件，test数据中最大运行31个周期，RUL中还有112个周期。故最大周期为143
truth_df.drop('more', axis=1, inplace=True)
test_df = test_df.merge(truth_df, on=['id'], how='left')
# 将test的最大周期这一列加到test_df中，各id在不同cycle的最大周期一致。
test_df['RUL'] = test_df['max'] - test_df['cycle']
# 算得test的实时RUL值
test_df.drop('max', axis=1, inplace=True)


# 删掉用来计算RUL的max这一列。
# 将RUL中大于130的值改为130
test_df['RUL'].loc[test_df['RUL'] >125]=125
# test 数据处理#######################################
#     print(train_df.shape)
#     print(test_df.shape)


# 将数据格式变为(样本循环次数, 时间窗大小：50, 特征数)
def gen_sequence(id_df, seq_length, seq_cols):
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        yield data_array[start:stop, :]


# 对应数据格式生成标签
def gen_labels(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements, :]


# 选择特征列
# sensor_cols = [ 's2', 's3','s4', 's6', 's7', 's8', 's9', 's11', 's12', 's13', 's14','s15', 's17', 's20', 's21']
# sequence_cols = ['setting1', 'setting2', 'cycle_norm']
# #     sequence_cols = ['setting1', 'setting2']
# sequence_cols.extend(sensor_cols)

# https://ieeexplore.ieee.org/document/10146287
sequence_cols = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
# sequence_cols = ['setting1', 'setting2', 'cycle_norm']
#     sequence_cols = ['setting1', 'setting2']
# sequence_cols.extend(sensor_cols)

# ******************************************************************测试集********************************************
length_of_test_df = []
for id in test_df['id'].unique():
    length_of_test = len(test_df[test_df['id'] == id])
    length_of_test_df.append(length_of_test)
# print(length_of_test_df)
id_of_test_df = list(range(1, 101))
# print(id_of_test_df)

# 按测试集长度大小排序的id号
length_of_test_df_sorted, id_of_test_df_sorted = zip(*sorted(zip(length_of_test_df, id_of_test_df)))

id_of_test_df_sorted_30 = id_of_test_df_sorted[:12]
id_of_test_df_sorted_60 = id_of_test_df_sorted[12:26]
id_of_test_df_sorted_90 = id_of_test_df_sorted[26:37]
id_of_test_df_sorted_120 = id_of_test_df_sorted[:]


# ******************************************************************测试集********************************************

# 数据集类
class MyDataFloder(Dataset):
    def __init__(self, x, y):
        self.input_feature = x
        self.input__label = y

    def __getitem__(self, index):
        return torch.from_numpy(self.input_feature[index]), self.input__label[index]

    def __len__(self):
        return len(self.input__label)


BATCH_SIZE_training = 128
BATCH_SIZE_test = 128

# for sequence_length in range(30,150,30):

for sequence_length in range(30, 60, 30):
    # 训练数据样本和标签################################
    seq_gen = (list(gen_sequence(train_df[train_df['id'] == id], sequence_length, sequence_cols))
               for id in train_df['id'].unique())
    locals()['training_sample_' + str(sequence_length)] = np.concatenate(list(seq_gen)).astype(np.float32)

    label_gen = [gen_labels(train_df[train_df['id'] == id], sequence_length, ['RUL'])
                 for id in train_df['id'].unique()]
    locals()['training_label_' + str(sequence_length)] = np.concatenate(label_gen).astype(np.float32)

    locals()['training_sample_' + str(sequence_length)] = locals()['training_sample_' + str(sequence_length)].transpose(
        0, 2, 1)
    # 训练数据样本和标签################################

    # 测试数据样本和标签################################
    test_sample = [test_df[test_df['id'] == id][sequence_cols].values[-sequence_length:]
                   for id in id_of_test_df_sorted_120]
    locals()['test_sample_120_' + str(sequence_length)] = np.asarray(test_sample).astype(np.float32)

    test_label = [test_df[test_df['id'] == id]['RUL'].values[-1] for id in id_of_test_df_sorted_120]
    locals()['test_label_120_' + str(sequence_length)] = np.array(test_label).astype(np.float32)

    locals()['test_sample_120_' + str(sequence_length)] = locals()['test_sample_120_' + str(sequence_length)].transpose(
        0, 2, 1)
    # 测试数据样本和标签################################

    # print("训练样本和标签：", locals()['training_sample_' + str(sequence_length)].shape,
    #       locals()['training_label_' + str(sequence_length)].shape)
    # print("测试样本和标签：", locals()['test_sample_120_' + str(sequence_length)].shape,
    #       locals()['test_label_120_' + str(sequence_length)].shape)
    #
    # locals()['train_dataset_' + str(sequence_length)] = MyDataFloder(
    #     locals()['training_sample_' + str(sequence_length)], locals()['training_label_' + str(sequence_length)])
    # locals()['test_dataset_' + str(sequence_length)] = MyDataFloder(locals()['test_sample_120_' + str(sequence_length)],
    #                                                                 locals()['test_label_120_' + str(sequence_length)])
    #
    # locals()['train_loader_' + str(sequence_length)] = torch.utils.data.DataLoader(
    #     dataset=locals()['train_dataset_' + str(sequence_length)], batch_size=BATCH_SIZE_training, shuffle=True)
    # locals()['test_loader_' + str(sequence_length)] = torch.utils.data.DataLoader(
    #     dataset=locals()['test_dataset_' + str(sequence_length)], batch_size=BATCH_SIZE_test, shuffle=False)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.recurrent import A3TGCN2

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # cuda or cpu
batch_size = 128

train_x_tensor = torch.from_numpy(training_sample_30).permute(0, 2, 1).contiguous().type(torch.FloatTensor).to(DEVICE)
train_target_tensor = torch.from_numpy(training_label_30).type(torch.FloatTensor).unsqueeze(2).to(DEVICE)

print("training samples: ", train_x_tensor.size(), train_target_tensor.size())


test_x_tensor = torch.from_numpy(test_sample_120_30).permute(0, 2, 1).contiguous().type(torch.FloatTensor).to(DEVICE)
test_target_tensor = torch.from_numpy(test_label_120_30).type(torch.FloatTensor).unsqueeze(1).unsqueeze(2).to(DEVICE)
print("test samples: ", test_x_tensor.size(), test_target_tensor.size())

# training samples:  torch.Size([17631, 14, 1, 30]) torch.Size([17631, 1])
# test samples:  torch.Size([63, 14, 1, 30]) torch.Size([63, 1])

train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
train_dataloader = torch.utils.data.DataLoader(train_dataset_new, batch_size=batch_size, shuffle=True,drop_last=False)

test_dataset_new = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
val_dataloader = torch.utils.data.DataLoader(test_dataset_new, batch_size=100, shuffle=False,drop_last=False)



import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.data_loader import Dataset_ECG, Dataset_Dhfm, Dataset_Solar, Dataset_Wiki
from model.FourierGNN import FGN
import time
import os
import numpy as np

import sys
print(sys.path)
from utils.utils import save_model, load_model, evaluate

# main settings can be seen in markdown file (README.md)
parser = argparse.ArgumentParser(description='fourier graph network for multivariate time series forecasting')
parser.add_argument('--data', type=str, default='RUL', help='data set')
parser.add_argument('--feature_size', type=int, default='14', help='feature size')
parser.add_argument('--seq_length', type=int, default=30, help='inout length')
parser.add_argument('--pre_length', type=int, default=1, help='predict length')
parser.add_argument('--embed_size', type=int, default=128, help='hidden dimensions')
parser.add_argument('--hidden_size', type=int, default=256, help='hidden dimensions')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='input data batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--exponential_decay_step', type=int, default=5)
parser.add_argument('--validate_freq', type=int, default=1)
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--train_ratio', type=float, default=0.7)
parser.add_argument('--val_ratio', type=float, default=0.2)
parser.add_argument('--device', type=str, default='cuda:0', help='device')

args = parser.parse_args()
print(f'Training configs: {args}')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = FGN(pre_length=args.pre_length,
            embed_size=args.embed_size,
            feature_size=args.feature_size,
            seq_length=args.seq_length,
            hidden_size=args.hidden_size).to(device)
my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.learning_rate, eps=1e-08)
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)
forecast_loss = nn.MSELoss(reduction='mean').to(device)


def validate(model, vali_loader):
    model.eval()
    cnt = 0
    loss_total = 0
    preds = []
    trues = []
    for i, (x, y) in enumerate(vali_loader):
        cnt += 1
        y = y.float().to(device)
        x = x.float().to(device)
        forecast = model(x)
        y = y.permute(0, 2, 1).contiguous()

        print(y.size(), forecast.size())
        loss = forecast_loss(forecast, y)
        loss_total += float(loss)
        forecast = forecast.detach().cpu().numpy()  # .squeeze()
        y = y.detach().cpu().numpy()  # .squeeze()
        preds.append(forecast)
        trues.append(y)
    preds = np.array(preds)
    trues = np.array(trues)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    score = evaluate(trues, preds)
    print(f'RAW : MAPE {score[0]:7.9%}; MAE {score[1]:7.9f}; RMSE {score[2]:7.9f}.')
    model.train()
    return loss_total/cnt

def test():
    result_test_file = 'output/'+args.data+'/train'
    model = load_model(result_test_file, 48)
    model.eval()
    preds = []
    trues = []
    sne = []
    for index, (x, y) in enumerate(val_dataloader):
        y = y.float().to(device)
        x = x.float().to(device)
        forecast = model(x)
        y = y.permute(0, 2, 1).contiguous()
        forecast = forecast.detach().cpu().numpy()  # .squeeze()
        y = y.detach().cpu().numpy()  # .squeeze()
        preds.append(forecast)
        trues.append(y)

    preds = np.array(preds)
    trues = np.array(trues)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    score = evaluate(trues, preds)
    print(f'RAW : MAPE {score[0]:7.9%}; MAE {score[1]:7.9f}; RMSE {score[2]:7.9f}.')

# create output dir
result_train_file = os.path.join('output', args.data, 'train')
result_test_file = os.path.join('output', args.data, 'test')
if not os.path.exists(result_train_file):
    os.makedirs(result_train_file)
if not os.path.exists(result_test_file):
    os.makedirs(result_test_file)


if __name__ == '__main__':

    for epoch in range(args.train_epochs):
        epoch_start_time = time.time()
        model.train()
        loss_total = 0
        cnt = 0
        for index, (x, y) in enumerate(train_dataloader):
            cnt += 1
            y = y.float().to(device)
            x = x.float().to(device)
            forecast = model(x)
            y = y.permute(0, 2, 1).contiguous()

            loss = forecast_loss(forecast, y)
            loss.backward()
            my_optim.step()
            loss_total += float(loss)

        if (epoch + 1) % args.exponential_decay_step == 0:
            my_lr_scheduler.step()
        if (epoch + 1) % args.validate_freq == 0:
            val_loss = validate(model, val_dataloader)

        print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f} | val_loss {:5.4f}'.format(
                epoch, (time.time() - epoch_start_time), loss_total / cnt, val_loss))
        save_model(model, result_train_file, epoch)



