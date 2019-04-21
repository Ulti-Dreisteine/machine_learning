# -*- coding: utf-8 -*-
"""
Created on Mon Feb 05 16:56:53 2018

@author: luolei

使用watermelon数据集进行逻辑回归
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
from scipy import interp  
from sklearn import preprocessing
from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold  

raw_data = pd.read_table(u'C:/Users/bbd/Desktop/exercise_1.txt', sep = ',')
#### functions ################################################################
def oneHotEncoding(X = list()):
    # 使用OneHotEncoding方式对X中的离散值元素进行编码
    labels = set(X)
    X_encoded = list()
    for i in X:
        p = np.zeros(len(labels))
        p[list(labels).index(i)] = 1
        X_encoded.append(p)
    return X_encoded, labels

def dateToFloat(date = str()):
    # 将表示时间的字符变量转化为连续型数值变量形式
    # 这里用的绝对时间；也可以考虑使用相对于某一时刻的时间差，解释性应该更强
    if len(date) != 10:
        print('时间数据的格式不合要求')
    else:
        return float(date[0:4]) * 365 + float(date[5:7]) * 30 + float(date[8:11]) * 1
#### data #####################################################################

#### main program #############################################################
#### 数据处理
# 标签
y_new = np.zeros([5000, 1])
for i in range(len(list(raw_data['company_type']))):
    if raw_data['company_type'][i] == '个体':
        y_new[i] = 1.0

# 测试集和训练集
index_names = list(raw_data.keys())[:-1]
X = list([raw_data['regcap_amount'], raw_data['esdate'], raw_data['ipo_company'], raw_data['company_industry']])

# 按照指标值的不同类型进行转码
X_new = list()

# 0. regcap_amount指标
regcap_amount = list(raw_data['regcap_amount'])
for i in range(len(raw_data['regcap_amount'])):
    if np.isnan(raw_data['regcap_amount'][i]):
        regcap_amount[i] = 0.0
X_new.append(np.array(regcap_amount))

# 1. estate指标
X_new.append([dateToFloat(i) for i in list(raw_data['esdate'])]) 

# 2. ipo_company指标
X_new.append(oneHotEncoding(list(raw_data['ipo_company']))[0])

# 3. company_industry指标
X_new.append(oneHotEncoding(list(raw_data['company_industry']))[0])

X_new = [np.array(i) for i in X_new]

#### 逻辑回归部分
## 数据准备
from sklearn.cross_validation import train_test_split
X_spl = np.hstack((X_new[0].reshape(5000, 1), X_new[1].reshape(5000, 1), X_new[2].reshape(5000, 1), X_new[3].reshape(5000, 20)))
# 数据标准化
X_norm = list()
for i in range(X_spl.shape[1]):
    l = [p[i] for p in X_spl]
    if min(l) == max(l):
        X_norm.append(np.array([i - i for i in l]))
    else:
        X_norm.append((l - min(l)) / (max(l) - min(l)))
X_norm = np.array(X_norm).T

y_spl = y_new


# 指标加权
X_weights = np.array([0.5, 0.125, 0.0, 0.0, 0.875, 0.0, 0.25, 0.75, 0.375, 0.625, 0.25, 0.625, 0.875, 0.875, 0.0, 0.5, 0.0, 0.0, 0.25, 0.875, 0.25, 0.0, 0.375])
X_w = list()
for i in range(len(X_weights)):
    X_w.append(np.array([p[i] for p in X_norm]) * X_weights[i])
X_w = np.array(X_w).T

####
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


## 进行Adaboost计算
# 使用6折交叉验证，并且画ROC曲线
cv = StratifiedKFold(y_spl.reshape(len(y_spl),), n_folds = 7)
y_spl = y_spl.reshape(len(y_spl),)
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 3), algorithm="SAMME.R", n_estimators = 10)
mean_tpr = 0.0  
mean_fpr = np.linspace(0, 1, 100)

for i, (train, test) in enumerate(cv):  
    #通过训练数据，使用Adaboost建立模型，并对测试集进行测试，求出预测得分  
    probas_ = clf.fit(X_w[train], y_spl[train]).predict_proba(X_w[test])     
    # Compute ROC curve and area the curve  
    # 通过roc_curve()函数，求出fpr和tpr，以及阈值  
    fpr, tpr, thresholds = metrics.roc_curve(y_spl[test], probas_[:, 1])  
    mean_tpr += interp(mean_fpr, fpr, tpr)          #对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数  
    mean_tpr[0] = 0.0                               #初始处为0  
    roc_auc = metrics.auc(fpr, tpr)  
    #画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来  
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    
#画对角线  
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  
  
mean_tpr /= len(cv)                     #在mean_fpr100个点，每个点处插值插值多次取平均  
mean_tpr[-1] = 1.0                      #坐标最后一个点为（1,1）  
mean_auc = metrics.auc(mean_fpr, mean_tpr)      #计算平均AUC值
#画平均ROC曲线  
#print mean_fpr,len(mean_fpr)  
#print mean_tpr  
plt.plot(mean_fpr, mean_tpr, 'k--',  
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)  
  
plt.xlim([-0.01, 1.01])  
plt.ylim([-0.01, 1.01])  
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')  
plt.title('Receiver operating characteristic example')  
plt.legend(loc="lower right")
plt.grid(True) 
plt.show()