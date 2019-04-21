# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 09:26:19 2018

@author: Dreisteine

local interpretation for GBDT model
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')
import copy
import matplotlib.pyplot as plt
import time

## 定义类和函数—————————————————————————————————————————————————————————————————————————————————————————————————————————
def generate_gbdt_model(X_train, y_train, n_estimators = 100, max_depth = 3):
    """
    根据数据生成一个GBDT模型
    """
    "数据缺失值补全"
    def fillNan(x):
        for column in x.columns:
            x[column].where(x[column].notnull(), 0.0, inplace = True)
    fillNan(X_train)
    fillNan(y_train)
    
    "数据格式转换"
    X_train = np.array(X_train).reshape(len(X_train), len(X_train.iloc[0]))
    y_train = np.array(y_train).reshape(len(y_train), 1)
    
    "GBDT学习"
    gbdt = GradientBoostingClassifier(n_estimators = n_estimators, 
                                      learning_rate = 0.01, 
                                      min_samples_split = 250,
                                      min_samples_leaf = 1,
                                      max_depth = max_depth, 
                                      subsample = 1,
                                      random_state = 10,
                                      loss = "deviance")
    gbdt.fit(X_train, y_train)
    return gbdt

class GetInformationOfTree(object):
    """
    获取一颗tree的节点和连接等信息
    """
    def __init__(self, tree):
        self.tree = tree
    
    def get_node_information(self):
        """
        获取节点相关基本信息
        """
        node_id_list = list(range(self.tree.tree_.node_count))
        left_children_id = list(self.tree.tree_.children_left)
        right_children_id = list(self.tree.tree_.children_right)
        
        "构建父子节点间的连接方式"
        parent_children_connections = dict()
        for node_id in node_id_list:
            parent_children_connections[node_id] = [left_children_id[node_id], right_children_id[node_id]]
        
        "记录每个节点上的样本数"
        n_sample_in_one_node = dict()
        for node_id in node_id_list:
            n_sample_in_one_node[node_id] = self.tree.tree_.n_node_samples[node_id]
        
        "记录每个节点的分离阈值"
        split_threshold_in_one_node = dict()
        for node_id in node_id_list:
            split_threshold_in_one_node[node_id] = self.tree.tree_.threshold[node_id]
        
        "记录每个节点的分裂特征"
        split_feature_in_one_node = dict()
        for node_id in node_id_list:
            split_feature_in_one_node[node_id] = self.tree.tree_.feature[node_id]
        
        return parent_children_connections, n_sample_in_one_node, split_threshold_in_one_node, split_feature_in_one_node
    
    def cal_pos_and_neg_ratio_in_each_node(self, X_train, y_train):
        """
        获取一颗树中所有节点上的正负节点数和比例
        X_train, y_train均为pd.DataFrame格式
        """
        t1 = time.time()
        pos_and_neg_counts_all_nodes = dict()
        node_id_list = list(range(self.tree.tree_.node_count))
        for node_id in node_id_list:
            pos_and_neg_counts_all_nodes[node_id] = {0 : 0, 1 : 0}
        
        "逐样本带入训练模型中, 计算每个节点上的"
        len_X_sample = len(X_train.iloc[0])
        X_train_array = np.array(X_train)
        y_train_array = np.array(y_train)
        
        train_sample = X_train_array.reshape(len(X_train_array), len_X_sample)
        train_label = [p[0] for p in y_train_array]
        decision_path_matrix = self.tree.decision_path(train_sample)
        
        "decision_path按照0-1标签样本进行划分"
        decision_path_matrix_1 = pd.DataFrame([decision_path_matrix[i].toarray()[0] for i in range(len(X_train_array)) if train_label[i] == 1])
        decision_path_matrix_0 = pd.DataFrame([decision_path_matrix[i].toarray()[0] for i in range(len(X_train_array)) if train_label[i] == 0])
        
        "列加和"
        decision_path_matrix_1_sum = decision_path_matrix_1.sum(axis = 0)
        decision_path_matrix_0_sum = decision_path_matrix_0.sum(axis = 0)
        
        pos_and_neg_counts_all_nodes = pd.concat([decision_path_matrix_0_sum, decision_path_matrix_1_sum], axis = 1)
        pos_and_neg_counts_all_nodes['node_id'] = pos_and_neg_counts_all_nodes.index
        "计算正样本比例"
        def cal_pos_ratio(x):
            return x[0] / (x[0] + x[1])
        pos_and_neg_counts_all_nodes['pos_ratio'] = pos_and_neg_counts_all_nodes[[0, 1]].apply(cal_pos_ratio, axis = 1)
        
        print('time cost for cal_pos_and_neg_ratio_in_each_node: %.4f second(s)' % (time.time() - t1))
        return pos_and_neg_counts_all_nodes    
    
    @staticmethod
    def cal_scores(x):
        "在这里定义score计算"
        return 4 * pow(x, 2) - 4 * x + 1
    
    def cal_LI_for_one_sample(self, X_train, y_train, test_sample):
        """
        计算一棵树上各特征的LI值
        """
        parent_children_connections, n_sample_in_one_node, split_threshold_in_one_node, split_feature_in_one_node = self.get_node_information()
        pos_and_neg_counts_all_nodes = self.cal_pos_and_neg_ratio_in_each_node(X_train, y_train)
        
        "首先区分叶节点, 并找出叶节点对应的分数值"
        scores = dict()
        leaf_nodes = []
        for key in parent_children_connections.keys():
            if parent_children_connections[key][0] == -1:
                leaf_nodes.append(key)
                scores[key] = pos_and_neg_counts_all_nodes.loc[key]['pos_ratio']
        
        "获得该节点i的决策路径和路径节点分裂对应特征"
        decision_path_matrix = self.tree.decision_path(test_sample)
        
        decision_path_node_series = []
        for i in range(decision_path_matrix.shape[1]):
            if decision_path_matrix[0, i] == 1:
                decision_path_node_series.append(i)
                
        decision_path_node_feature_ids = []
        for i in decision_path_node_series[: -1]:
            decision_path_node_feature_ids.append(self.tree.tree_.feature[i])
            
        "反向求解决策路径中个节点score以及对应的特征id等"
        "初始化"
        node_score_and_id = dict()
        decision_path_node_series.reverse()
        node_score_and_id[decision_path_node_series[0]] = dict()
        node_score_and_id[decision_path_node_series[0]]['split_feature_id'] = ''
        node_score_and_id[decision_path_node_series[0]]['scores'] = self.cal_scores(pos_and_neg_counts_all_nodes.loc[decision_path_node_series[-1]]['pos_ratio'])
        
        for i in range(1, len(decision_path_node_series)):
            parent_node = decision_path_node_series[i]
            parent_children_connections_copy = copy.deepcopy(parent_children_connections)
            another_child_node = [p for p in parent_children_connections_copy[parent_node] if p != decision_path_node_series[i - 1]][0]
            node_score_and_id[decision_path_node_series[i]] = dict()
            node_score_and_id[decision_path_node_series[i]]['split_feature_id'] = split_feature_in_one_node[parent_node]
            
            N1 = pos_and_neg_counts_all_nodes.loc[decision_path_node_series[i - 1]][[0, 1]].sum()
            N2 = pos_and_neg_counts_all_nodes.loc[another_child_node][[0, 1]].sum()
            
            
            "注意论文里每个node的scores是怎么定义的"
            another_child_ratio = pos_and_neg_counts_all_nodes.loc[another_child_node]['pos_ratio']
            node_score_and_id[decision_path_node_series[i]]['scores'] = (N1 * node_score_and_id[decision_path_node_series[i - 1]]['scores'] + N2 * self.cal_scores(another_child_ratio)) / (N1 + N2)
               
        "在这颗树里计算LI"
        decision_path_node_series.reverse()
        feature_LI_dict = dict()
        for i in decision_path_node_series[: -1]:
            if node_score_and_id[i]['split_feature_id'] in feature_LI_dict.keys():
                feature_LI_dict[node_score_and_id[i]['split_feature_id']] += node_score_and_id[\
                               decision_path_node_series[decision_path_node_series.index(i) + 1]]['scores'] - node_score_and_id[i]['scores']
            else:   
                feature_LI_dict[node_score_and_id[i]['split_feature_id']] = dict()
                feature_LI_dict[node_score_and_id[i]['split_feature_id']] = node_score_and_id[\
                               decision_path_node_series[decision_path_node_series.index(i) + 1]]['scores'] - node_score_and_id[i]['scores']
                
        return decision_path_node_feature_ids, feature_LI_dict

class CalFeatureContributionsOfGBDT(object):
    """
    计算GBDT模型对于某测试样本的的FC值
    """
    def __init__(self, gbdt, X_train, y_train, test_sample):
        self.gbdt = gbdt
        self.X_train = X_train
        self.y_train = y_train
        self.test_sample = test_sample
    
    def _get_all_trees_of_gbdt(self):
        """
        获得GBDT模型上的所有树
        """
        gbdt_trees = []
        for i in range(self.gbdt.n_estimators):
            gbdt_trees.append(self.gbdt[i, 0])
        self.gbdt_trees = gbdt_trees
    
    def _get_tree_weights(self):
        '获取每棵树的权重使用每个样本在每棵树上预测值的残差平方和进行加权'
        y_pred_each_tree = list()
        for i in range(self.gbdt.n_estimators):
            clf = self.gbdt[i, 0]
            y_pred_each_tree.append(clf.predict(self.X_train))
            
        self.tree_weights = list()
        for i in range(len(y_pred_each_tree)):
            s = 0
            for j in range(len(y_pred_each_tree[i])):
                s += pow(y_pred_each_tree[i][j], 2)
            self.tree_weights.append(s)
        self.tree_weights = [p / sum(self.tree_weights) for p in self.tree_weights]
        
    def cal_LI_for_all_trees(self):
        self._get_all_trees_of_gbdt()
        tree_LI_list = []
        for tree in self.gbdt_trees:
            t2 = time.time()
            print('calculating LI values: processing tree %s' % str(self.gbdt_trees.index(tree)))
            git = GetInformationOfTree(tree)
            tree_LI_list.append(git.cal_LI_for_one_sample(self.X_train, self.y_train, self.test_sample))
            print('time cost for feature LI vaue  of this tree: %.4f second(s)' % (time.time() - t2))
            print(' ')
        self.tree_LI_list = tree_LI_list
        return self.tree_LI_list
    
    def cal_FC(self):
        self._get_all_trees_of_gbdt()
        self._get_tree_weights()
        self.tree_LI_list = self.cal_LI_for_all_trees()
        
        "对GBDT上每棵树上各特征LI值按照计算出来的weights进行加权"
        for i in range(self.gbdt.n_estimators):
            for key in self.tree_LI_list[i][1].keys():
                self.tree_LI_list[i][1][key] *= self.tree_weights[i]
        
        "GBDT上所有树的特征LI值字典合并"
        self.FC_table = pd.DataFrame([p[1] for p in self.tree_LI_list])
        self.FC_table = self.FC_table.where(self.FC_table.notnull(), 0.0)
        self.FC_table = pd.DataFrame(self.FC_table.sum(), columns = ['FC_value'])
        return self.FC_table
    
## 主程序——————————————————————————————————————————————————————————————————————————————————————————————————————————————
if __name__ == '__main__':
    ## part 1. 输入数据------------------------------------------------------------------------------------------------
    "指定训练集和测试集"
    X_train = pd.read_csv('X_train.csv', encoding = 'utf-8', sep = ',')
    y_train = pd.read_csv('y_train.csv', encoding = 'utf-8', sep = ',')
    
    "指定特定个体"
    test_sample = np.array(X_train.loc[6000]).reshape(1, len(X_train.iloc[0]))
    
    ## part 2. 获得待分析模型-------------------------------------------------------------------------------------------
    gbdt = generate_gbdt_model(X_train, y_train)
    
    ## part 3. 单独对一棵树的特征LI值计算--------------------------------------------------------------------------------
    tree = gbdt[17, 0]
    git = GetInformationOfTree(tree)
    feature_LI_list = git.cal_LI_for_one_sample(X_train, y_train, test_sample)
    
    ## part 4. 计算GBDT模型的FC值---------------------------------------------------------------------------------------
    time_start = time.time()
    cfc = CalFeatureContributionsOfGBDT(gbdt, X_train, y_train, test_sample)
    FC_table = cfc.cal_FC()
    FC_table = FC_table.sort_values('FC_value', ascending = False)
    print('TOTAL TIME COST FOR LOCAL INTERPRETATION: %.4f second(s)' % (time.time() - time_start))
    
    ## part 5. 结果展示-------------------------------------------------------------------------------------------------
    plt.figure(figsize = [8, 5])
    plt.suptitle('Local Interpretation Results for GBDT', x = 0.55, y = 1)
    plt.subplot(2, 1, 1)
    plt.plot(cfc.tree_weights)
    plt.xlabel('tree label')
    plt.ylabel('weight of tree')
    plt.grid(True)
    plt.tight_layout()
    
    plt.subplot(2, 1, 2)
    plt.bar(range(len(FC_table)), FC_table['FC_value'])
    plt.xticks(range(len(FC_table)), ['feature_' + str(p) for p in FC_table.index], rotation = 90)
    plt.ylabel('FC value')
    plt.grid(True)
    plt.tight_layout()