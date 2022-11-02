# 开发人员：纪张逸凡
# 开发时间：2021/7/6 15:59
import torch
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
import time
import os
import seaborn as sns
from visdom import Visdom
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from model import LstmModel
from model_1 import gruModel
from model_with_self_attention import GRUModel
from sklearn.metrics import confusion_matrix
time_start = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Nback作为训练集，Stew作为测试集

# Hyper-parameters

input_size = 14
hidden_size = 64
num_layers = 2
num_classes = 2  #识别类别数量
batch_size = 512  #train_dataset里面的，批的大小
num_epochs = 10  #迭代次数
learning_rate = 0.01 #学习率

seed = 42
dt = pd.read_csv("nb-α.csv", header=None)
dataset = dt.values
X_train = dataset[:, :input_size]
Y_train = dataset[:, input_size]
dt = pd.read_csv("st-α.csv", header=None)
dataset = dt.values
X_test = dataset[:, :input_size]
Y_test = dataset[:, input_size]
F = []
F1 = []
SPE = []
SEN = []
#print(X)
#print(Y)
from sklearn.preprocessing import StandardScaler
X_train = StandardScaler().fit_transform(X_train)   #将数据转化为0,1正态分布
X_train = X_train.reshape(-1, 5, X_train.shape[1])
encoder = LabelEncoder()
encoder.fit(Y_train)
Y_train = encoder.transform(Y_train)
Y_train = Y_train[::5]
X_test = StandardScaler().fit_transform(X_test)
X_test = X_test.reshape(-1, 5, X_test.shape[1])
encoder = LabelEncoder()
encoder.fit(Y_test)
Y_test = encoder.transform(Y_test)
Y_test = Y_test[::5]
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
X_train, Y_train = torch.FloatTensor(X_train), torch.LongTensor(Y_train)
X_test, Y_test = torch.FloatTensor(X_test), torch.LongTensor(Y_test)
train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


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



model = GRUModel(input_size, hidden_size, num_layers, num_classes).to(device)
# model = gruModel(input_size, hidden_size, num_layers, num_classes).to(device)
# model = LstmModel(input_size, hidden_size, num_layers, num_classes).to(device)


## Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
# viz = Visdom()
# viz.line([[0., 0.]], [0], win='train', opts=dict(title='Train: N-back.', legend=['loss', 'acc']))
for epoch in range(15):
    model.train()
    for i, (images, labels) in enumerate(train_loader):#test_loader
        #images = images.reshape(-1, 64, input_size).to(device)

        # images = images.reshape(images.shape[0],1,input_size).to(device)
        images = images.to(device)
        # images2 = images.reshape(images.shape[0],1, input_size).to
        # images = torch.cat([images1,images2],1)
        # print(images)
        # print(images.shape)
        labels = labels.to(device)
        # print(labels)
        # Forward pass
        # outputs = model(images)
        outputs, attention_weight = model(images)
        # attention = np.average(attention_weight.cpu().detach().numpy(), axis=0)
        # df = pd.DataFrame(attention)
        # corr = df.corr()
        # sns.heatmap(corr, cmap='Reds', annot=True)
        # plt.show()

        # outputs = outputs.reshape(9,-1)
        # labels = labels.reshape(9,-1)
        # print(outputs.shape)
        # print(labels.shape)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
               .format(epoch+1, num_epochs, i+1, total_step, loss.item()))





        # _, predicted = torch.max(outputs.data, 1)
        # train_acc = ((predicted == labels).sum().item()) / labels.size(0)
        #
        # loss1 = loss.item()
        # acc = train_acc
        # viz.line([[loss1, acc]], [epoch], win='train', update='append')

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    y = np.zeros([2, 2])
    c = []
    d = []
    for images, labels in test_loader:#train_loader
        # print(images.shape[0])
        images = images.to(device)
        labels = labels.to(device)
        # label_record = (labels)
        # print(label_record)
        d.append(labels[0].tolist())
        # outputs = model(images)
        outputs, attention_weight = model(images)

        # attention = np.average(attention_weight.cpu().detach().numpy(), axis=0)
        # df = pd.DataFrame(attention)
        # corr = df.corr()
        # sns.heatmap(corr, cmap='Blues', annot=True)
        # plt.show()

        _, predicted = torch.max(outputs.data, 1)
        # predicted_record = (predicted)
        # print(predicted_record)

        c.append(predicted[0].tolist())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        c_matrix = confusion_matrix(np.array(labels.cpu()), np.array(predicted.cpu()))
        y = y + c_matrix
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Test Accuracy of the model on the test : {} '.format(1 * correct / total))

print(y)
TP = y[0][0]
FN = y[0][1]
FP = y[1][0]
TN = y[1][1]
# print(TP)
acc = (TP + TN) / (TP + TN + FP + FN)
pre = TP / (TP + FP)  # 精确度
npv = TN / (TN + FN)  # 召回率
spe = TN / (TN + FP)
sen = TP / (TP + FN)
# f11 = (2 * pre * sen) / (pre + sen)
# f12 = (2 * npv * spe) / (npv + spe)
f1 = (2*(sen+spe)/2*(pre+npv)/2)/((sen+spe)/2+(pre+npv)/2)
a1 = FP / (FP + TN)
AUC = (1+spe - a1)/2
print('acc : {}'.format(round(acc, 4)))
print('f1 score : {}'.format(round(f1, 4)))
print('sen : {}'.format(round(sen, 4)))
print('spe : {}'.format(round(spe, 4)))
print('auc : {}'.format(round(AUC, 4)))


        #print(c_matrix)
        # 3
        # corrects = c_matrix.diagonal(offset=0)  # 抽取对角线的每种分类的识别正确个数
        # per_kinds = c_matrix.sum(axis=1)  # 抽取每个分类数据总的测试条数
        # test_num = 1440 / batch_size
        # print("混淆矩阵总元素个数：{0},测试集总个数:{1}".format(int(np.sum(c_matrix)), test_num))
        # print(c_matrix)
        #
        # # 获取每种Emotion的识别准确率
        # print("每种情感总个数：", per_kinds)
        # print("每种情感预测正确的个数：", corrects)
        # print("每种情感的识别准确率为：{0}".format([rate * 100 for rate in corrects / per_kinds]))

        # # 绘制混淆矩阵
        # Workload = 2  # 这个数值是具体的分类数，大家可以自行修改
        # labels = ['LCW', 'HCW']  # 每种类别的标签
        #
        # # 显示数据
        # plt.imshow(c_matrix, cmap=plt.cm.Blues)
        # plt.colorbar()
        # # 在图中标注数量/概率信息
        # thresh = c_matrix.max() / 2  # 数值颜色阈值，如果数值超过这个，就颜色加深。
        # for x in range(Workload):
        #     for y in range(Workload):
        #         # 注意这里的matrix[y, x]不是matrix[x, y]
        #         info = int(c_matrix[y, x])
        #         plt.text(x, y, info,
        #                  verticalalignment='center',
        #                  horizontalalignment='center',
        #                  color="white" if info > thresh else "black")
        #
        # plt.tight_layout()  # 保证图不重叠
        # plt.yticks(range(Workload), labels)
        # plt.xticks(range(Workload), labels, rotation=45)  # X轴字体倾斜45°
        # plt.show()
        # plt.close()
        # 3
    #     TP = c_matrix[0][0]
    #     FP = c_matrix[0][1]
    #     FN = c_matrix[1][0]
    #     TN = c_matrix[1][1]
    #     acc = (TP + TN) / (TP + TN + FP + FN)
    #     P1 = TP / (TP + FP)  # 精确度
    #     R1 = TP / (TP + FN)  # 召回率
    #     f11 = (2 * P1 * R1) / (P1 + R1)
    #     P2 = TN / (TN + FP)  # 精确度
    #     R2 = TN / (TN + FN)  # 召回率
    #     f12 = (2 * P2 * R2) / (P2 + R2)
    #     f1 = (f11 + f12) / 2
    #     sen = TP / (TP + FN)
    #     spe = TN / (TN + FP)
    #     F1.append(f1)
    #     SPE.append(spe)
    #     SEN.append(sen)
    #     print('Test Accuracy of the model on the 40 test : {} '.format(1 * correct / total))
    #     f = 1 * correct / total
    #     F.append(f)
    # acc = np.mean(F)
    # print('Mean Test Accuracy of the model :{} '.format(round(acc, 4)))
    # F1_1 = np.mean(F1)
    # SPE_1 = np.mean(spe)
    # SEN_1 = np.mean(sen)
    # AUC = (SEN_1 + SPE_1)/2
    # print('f1 score : {}'.format(round(F1_1, 4)))
    # print('sen : {}'.format(round(SEN_1, 4)))
    # print('spe : {}'.format(round(SPE_1, 4)))
    # print('auc : {}'.format(round(AUC, 4)))
    # time_end = time.time()
    # abb = time_end-time_start
    # print('Time of code running: {} s'.format(abb))


