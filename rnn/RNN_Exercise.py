# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 12:25:45 2018

@author: Dreisteine
"""
import numpy as np
#import tensorflow as tf
import matplotlib.pyplot as plt

#### functions ################################################################
def genRawData(size = 1000):
    '''生成数据
    输入数据X：在时间t，Xt的值有50%的概率为1，50%的概率为0；
    输出数据Y：在实践t，Yt的值有50%的概率为1，50%的概率为0，除此之外，
    如果`Xt-3 == 1`，Yt为1的概率增加50%， 如果`Xt-8 == 1`，则Yt为1的概率减少25%， 如果上述两个条件同时满足，则Yt为1的概率为75%。
    '''
    raw_x = np.array(np.random.choice(2,size=(size,)))
    raw_y = []
    for i in range(size):
        threshold = 0.5
        if raw_x[i-3] == 1:
            threshold += 0.5
        if raw_x[i-8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            raw_y.append(0)
        else:
            raw_y.append(1)
    return raw_x, raw_y

def lossFunc(y_predic = list(), y_actual = list):
    '''
    使用平方差之和作为累计损失函数
    '''
    d = 0
    for i in range(len(y_predic)):
        d += pow(y_predic[i] - y_actual[i], 2)
    return d

def cellOutput(xt, st1, U, W, V):
    '''
    此处st1指上个cell传递下来的s参数
    '''
    st = np.tanh(np.dot(U, xt) + np.dot(W, st1))
    ot = softmax(np.dot(V, st))
    return ot, st

def softmax(x):
    '''
    返回x列表中每个元素对应的softmax值
    '''
    return np.exp(x) / np.sum(np.exp(x),axis = 0)
 
#### main program #############################################################
#### 产生数据并将数据进行处理以输入RNN网络中
## raw_data是使用gen_data()函数生成的数据，分别是raw_x和raw_y
raw_x, raw_y = genRawData(size = 1000)
data_length = len(raw_x)

## 首先将数据切分成batch_size份，0-batch_size，batch_size-2*batch_size。。。
batch_size = 1
num_steps = 5
batch_partition_length = data_length // batch_size
data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
for i in range(batch_size):
    data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
    data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]

## 因为RNN模型一次只处理num_steps个数据，所以将每个batch_size再进行切分成epoch_size份，每份num_steps个数据。注意这里的epoch_size和模型训练过程中的epoch不同。 
epoch_size = batch_partition_length // num_steps

## x是0-num_steps， batch_partition_length -batch_partition_length +num_steps。。。共batch_size个
## RNN的每批输入为x[i][0],为长度为num_steps的向量
x = list()
y = list()
for i in range(epoch_size):
    x.append(data_x[:, i * num_steps:(i + 1) * num_steps])
    y.append(data_y[:, i * num_steps:(i + 1) * num_steps])
    
#### 网络初始化和训练 #########################################################
num_hl = 3
U = np.zeros([num_hl, num_steps])
W = np.zeros([num_hl, num_hl])
V = np.zeros([num_steps, num_hl])

## 每一条数据训练的输出

