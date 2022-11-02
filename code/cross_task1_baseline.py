# 开发人员：纪张逸凡
# 开发时间：2021/7/6 15:59
import torch
import time
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from model import LstmModel
from model_with_self_attention import GRUModel
from sklearn.metrics import confusion_matrix
time_start = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Stew作为训练集，Nback作为测试集

# Hyper-parameters

input_size = 168
hidden_size = 128
num_layers = 1
num_classes = 2  #识别类别数量
batch_size = 256 #train_dataset里面的，批的大小
num_epochs = 10  #迭代次数
learning_rate = 0.001#学习率

seed = 42
dt = pd.read_csv("st_all.csv", header=None)
dataset = dt.values
X_train = dataset[:, :input_size]
Y_train = dataset[:, input_size]
dt = pd.read_csv("nb_all.csv", header=None)
dataset = dt.values
X_test = dataset[:, :input_size]
Y_test = dataset[:, input_size]
F = []
#print(X)
#print(Y)
from sklearn.preprocessing import StandardScaler
X_train = StandardScaler().fit_transform(X_train)   #将数据转化为0,1正态分布
#X_train = X_train.reshape(-1, 5, X_train.shape[1])
encoder = LabelEncoder()
encoder.fit(Y_train)
Y_train = encoder.transform(Y_train)
#Y_train = Y_train[::5]
X_test = StandardScaler().fit_transform(X_test)
#X_test = X_test.reshape(-1, 5, X_test.shape[1])
encoder = LabelEncoder()
encoder.fit(Y_test)
Y_test = encoder.transform(Y_test)
#Y_test = Y_test[::5]
X_train, Y_train = torch.FloatTensor(X_train), torch.LongTensor(Y_train)
X_test, Y_test = torch.FloatTensor(X_test), torch.LongTensor(Y_test)


def seed_torch(seed=seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
torch.backends.cudnn.enabled = True  #pytorch 使用CUDANN 加速，即使用GPU加速
torch.backends.cudnn.benchmark = False  # cuDNN使用的非确定性算法自动寻找最适合当前配置的高效算法，设置为False 则每次的算法一致
torch.backends.cudnn.deterministic = True  # 设置每次返回的卷积算法是一致的
torch.manual_seed(seed)  # 为当前CPU 设置随机种子
torch.cuda.manual_seed(seed)  # 为当前的GPU 设置随机种子
torch.cuda.manual_seed_all(seed)  # 当使用多块GPU 时，均设置随机种子
seed_torch(seed)

classifier = LogisticRegression()
# classifier = GaussianNB()
# classifier = RandomForestClassifier() #基线随机森林
# classifier = neighbors.KNeighborsClassifier(algorithm='auto') #基线knn
# classifier = svm.SVC(C=2, kernel='linear', decision_function_shape='ovo') #基线svm
classifier.fit(X_train, Y_train.ravel())
tra_label = classifier.predict(X_train)  # 训练集的预测标签
tes_label = classifier.predict(X_test)

print("训练集：", accuracy_score(Y_train, tra_label))
print("acc：", accuracy_score(Y_test, tes_label))
c_matrix = confusion_matrix(np.array(Y_test), np.array(tes_label))
# print(c_matrix)
TP = c_matrix[0][0]
FN = c_matrix[0][1]
FP = c_matrix[1][0]
TN = c_matrix[1][1]
acc = (TP + TN) / (TP + TN + FP + FN)
pre = TP / (TP + FP)  # 精确度
npv = TN / (TN + FN)  # 召回率
spe = TN / (TN + FP)
sen = TP / (TP + FN)
f1 = (2*(sen+spe)/2*(pre+npv)/2)/((sen+spe)/2+(pre+npv)/2)
a1 = FP / (FP + TN)
AUC = (1+spe - a1)/2
print('acc : {}'.format(round(acc, 4)))
print('f1 score : {}'.format(round(f1, 4)))
print('sen : {}'.format(round(sen, 4)))
print('spe : {}'.format(round(spe, 4)))
print('auc : {}'.format(round(AUC, 4)))
time_end = time.time()
abb = time_end-time_start
print('Time of code running: {} s'.format(abb))


