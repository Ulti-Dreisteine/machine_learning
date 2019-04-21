# -*- coding: utf-8 -*-
"""
Created on Mon Feb 05 16:56:53 2018

@author: luolei

建立了三个模型: GBDT, XGBoost和LSTM
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import seaborn as sns
import random
import time
sns.set_style('whitegrid')

## ————————————————————————————————————————————————————————————————————————————
## functions ——————————————————————————————————————————————————————————————————
## ————————————————————————————————————————————————————————————————————————————
def MAPE(y_test, y_predicted):
    '''
    计算平均绝对百分比误差
    '''
    epsilon = 0.001
    if (type(y_test) == np.ndarray) & (type(y_predicted) == np.ndarray):
        y_test = epsilon +  y_test
        y_predicted = [p[0] for p in list(y_predicted)]
        return np.mean([abs((y_predicted[i] - y_test[i]) / y_test[i]) for i in range(len(y_test))])
    else:
        
        y_test = [p + epsilon for p in list(y_test[list(y_test.columns)[0]])]
        y_predicted = list(y_predicted)
        return np.mean([abs((y_predicted[i] - y_test[i]) / y_test[i]) for i in range(len(y_test))])
    
## ————————————————————————————————————————————————————————————————————————————
## data ———————————————————————————————————————————————————————————————————————
## ————————————————————————————————————————————————————————————————————————————
data = pd.read_csv('test_data.csv', sep = ',')[['time', 'KWH']]
data['KWH'] = 2 * (data['KWH'] - data['KWH'].min()) / (data['KWH'].max() - data['KWH'].min()) - 1
## 数据作图 
plt.figure(figsize = [6, 3])
plt.plot(data['KWH'])

## 构造样本
sample = list()
for i in range(len(data) - 1):
    sample.append([data.iloc[i]['KWH'], data.iloc[i + 1]['KWH']])
sample = pd.DataFrame(sample)

## 划分训练集和测试集
train_set_ratio = 0.7
train_set_index_list = random.sample(list(range(len(sample))), int(len(sample) * train_set_ratio))
test_set_index_list = list(set(range(len(sample))).difference(train_set_index_list))

train_sample = pd.DataFrame(sample.loc[train_set_index_list])
test_sample = pd.DataFrame(sample.loc[test_set_index_list])

X_train = pd.DataFrame(train_sample[0]).reset_index(drop = True)
y_train = pd.DataFrame(train_sample[1]).reset_index(drop = True)
X_test = pd.DataFrame(test_sample[0]).reset_index(drop = True)
y_test = pd.DataFrame(test_sample[1]).reset_index(drop = True)

## ————————————————————————————————————————————————————————————————————————————
## GBDT 模型 ——————————————————————————————————————————————————————————————————
## ————————————————————————————————————————————————————————————————————————————
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn import metrics
mape = dict()
gbdt_model = GradientBoostingRegressor(n_estimators = 300, 
                                        learning_rate = 0.01, 
                                        min_samples_split = 100, 
                                        min_samples_leaf = 200, 
                                        max_depth = 4, 
                                        subsample = 1, 
                                        random_state = 10, 
                                        loss = "lad")
gbdt_model.fit(X_train, y_train)
y_predicted = gbdt_model.predict(X_test)

## 预测结果与实际测试集结果对比
plt.figure(figsize = [6, 4])
plt.plot(y_test)
plt.hold(True)
plt.plot(y_predicted, c = 'r', alpha = 0.6)
plt.legend(['true', 'predicted'])
plt.xlabel('sample number')
plt.ylabel('KWH')
plt.title('GBDT model')

## 计算平均绝对百分比误差
mape['gbdt'] = MAPE(y_test, y_predicted)

## ————————————————————————————————————————————————————————————————————————————
## XGBOOST 模型 ———————————————————————————————————————————————————————————————
## ————————————————————————————————————————————————————————————————————————————
from xgboost.sklearn import XGBRegressor
xgb_model = XGBRegressor(learning_rate = 0.1,
                     n_estimators = 100,
                     seed = 0,
                     loss = 'mse')
xgb_model.fit(X_train, y_train)
y_predicted = xgb_model.predict(X_test)

## 预测结果与实际测试集结果对比
plt.figure(figsize = [6, 4])
plt.plot(y_test)
plt.hold(True)
plt.plot(y_predicted, c = 'r', alpha = 0.6)
plt.legend(['true', 'predicted'])
plt.xlabel('sample number')
plt.ylabel('KWH')
plt.title('XGBoost model')

## 计算平均绝对百分比误差
mape['xgboost'] = MAPE(y_test, y_predicted)

## ————————————————————————————————————————————————————————————————————————————
## LSTM 模型 ——————————————————————————————————————————————————————————————————
## ————————————————————————————————————————————————————————————————————————————
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

def build_model(seq_len):
    model = Sequential()

    model.add(LSTM(1, input_dim = 1, input_length = seq_len, return_sequences = False))
    
    start = time.time()
    model.compile(loss = "mse", optimizer = "rmsprop")
    print("Compilation Time : ", time.time() - start)
    return model

'样本迭代次数'
epochs  = 1000
'预测序列长度'
seq_len = 10

time_series = list(data['KWH'])

'样本序列归一化'
time_series = 1 * ((time_series - np.min(time_series)) / (np.max(time_series) - np.min(time_series))) - 0

## 构造训练集和测试集
train_set_head_num_list = random.sample(list(range(len(time_series) - seq_len)), 
                                        int((len(time_series) - seq_len) * train_set_ratio))
test_set_head_num_list = list(set(range(len(time_series) - seq_len))\
                              .difference(train_set_head_num_list))

train_set = list()
for i in train_set_head_num_list:
    train_set.append(time_series[i : i + seq_len + 1])
    
test_set = list()
for i in test_set_head_num_list:
    test_set.append(time_series[i : i + seq_len + 1])

X_train = np.array([p[: -1] for p in train_set])
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
y_train = np.array([p[-1] for p in train_set])
y_train = np.reshape(y_train, (y_train.shape[0],))

X_test = np.array([p[: -1] for p in test_set])
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_test = np.array([p[-1] for p in test_set])
y_test = np.reshape(y_test, (y_test.shape[0],))

model = build_model(seq_len)
model.fit(X_train, y_train, batch_size = 3000, nb_epoch = epochs)
y_predicted = model.predict(X_test)

## 预测结果与实际测试集结果对比
plt.figure(figsize = [6, 4])
plt.plot(y_test)
plt.hold(True)
plt.plot(y_predicted, c = 'r', alpha = 0.5)
plt.legend(['true', 'predicted'])
plt.xlabel('sample number')
plt.ylabel('KWH')
plt.title('LSTM model')

## 计算平均绝对百分比误差
mape['LSTM'] = MAPE(y_test, y_predicted)
