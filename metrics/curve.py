import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import cohen_kappa_score, accuracy_score


# fpr0, tpr0, thresholds0 = roc_curve(y_true0,y_sore0)
# fpr1, tpr1, thresholds1 = roc_curve(y_true1,y_sore1)
# fpr2, tpr2, thresholds2 = roc_curve(y_true2,y_sore2)
# fpr3, tpr3, thresholds3 = roc_curve(y_true3,y_sore3)
# fpr4, tpr4, thresholds4 = roc_curve(y_true4,y_sore4)


# roc_auc0 = auc(fpr0, tpr0)
# roc_auc1 = auc(fpr1, tpr1)
# roc_auc2 = auc(fpr2, tpr2)
# roc_auc3 = auc(fpr3, tpr3)
# roc_auc4 = auc(fpr4, tpr4)
#
# plt.title('Receiver Operating Characteristic')
# plt.rcParams['figure.figsize'] = (10.0, 10.0)
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus']=False
# # 设置标题大小
# plt.rcParams['font.size'] = '16'
# plt.plot(fpr0, tpr0, 'k-',color='k',linestyle='-.',linewidth=3,markerfacecolor='none',label=u'AA_AUC = %0.5f'% roc_auc0)
# plt.plot(fpr1, tpr1, 'k-',color='grey',linestyle='-.',linewidth=3,label=u'A_AUC = %0.5f'% roc_auc1)
# plt.plot(fpr2, tpr2, 'k-',color='r',linestyle='-.',linewidth=3,markerfacecolor='none',label=u'B_AUC = %0.5f'% roc_auc2)
# plt.plot(fpr3, tpr3, 'k-',color='red',linestyle='-.',linewidth=3,markerfacecolor='none',label=u'C_AUC = %0.5f'% roc_auc3)
# plt.plot(fpr4, tpr4, 'k-',color='y',linestyle='-.',linewidth=3,markerfacecolor='none',label=u'D_AUC = %0.5f'% roc_auc4)
#
# plt.legend(loc='lower right')
# plt.plot([0,1],[0,1],'r--')
# plt.xlim([-0.1,1.1])
# plt.ylim([-0.1,1.1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.grid(linestyle='-.')
# plt.grid(True)
# plt.show()
#
#
# y_test_all = label_binarize(true_labels_i, classes=[0,1,2,3,4])
#
# y_score_all=test_Y_i_hat
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(len(classes)):
#     fpr[i], tpr[i], thresholds = roc_curve(y_test_all[:, i],y_score_all[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
#
#
# # micro-average ROC curve（方法一）
# fpr["micro"], tpr["micro"], thresholds = roc_curve(y_test_all.ravel(),y_score_all.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
# # macro-average ROC curve 方法二）
#
# all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
#
# mean_tpr = np.zeros_like(all_fpr)
# for i in range(len(classes)):
#     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# # 求平均计算ROC包围的面积AUC
# mean_tpr /= len(classes)
# fpr["macro"] = all_fpr
# tpr["macro"] = mean_tpr
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#
# #画图部分
# plt.figure()
# plt.plot(fpr["micro"], tpr["micro"],'k-',color='y',
#          label='XXXX ROC curve micro-average(AUC = {0:0.4f})'
#                ''.format(roc_auc["micro"]),
#           linestyle='-.', linewidth=3)
#
# plt.plot(fpr["macro"], tpr["macro"],'k-',color='k',
#          label='XXXX ROC curve macro-average(AUC = {0:0.4f})'
#                ''.format(roc_auc["macro"]),
#           linestyle='-.', linewidth=3)
# plt.plot([0,1],[0,1],'r--')
# plt.xlim([-0.1,1.1])
# plt.ylim([-0.1,1.1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.legend(loc="lower right")
# plt.grid(linestyle='-.')
# plt.grid(True)
# plt.show()



# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
from sklearn.preprocessing import label_binarize

if __name__ == '__main__':
    np.random.seed(0)
    data = pd.read_csv('iris.list', header=None)  # 读取数据
    iris_types = data[4].unique()
    n_class = iris_types.size
    x = data.iloc[:, :2]  # 只取前面两个特征
    y = pd.Categorical(data[4]).codes  # 将标签转换0,1,...
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, random_state=0)
    y_one_hot = label_binarize(y_test, np.arange(n_class))  # 装换成类似二进制的编码
    alpha = np.logspace(-2, 2, 20)  # 设置超参数范围
    model = LogisticRegressionCV(Cs=alpha, cv=3, penalty='l2')  # 使用L2正则化
    model.fit(x_train, y_train)
    print
    '超参数：', model.C_
    # 计算属于各个类别的概率，返回值的shape = [n_samples, n_classes]
    y_score = model.predict_proba(x_test)
    # 1、调用函数计算micro类型的AUC
    print
    '调用函数auc：', metrics.roc_auc_score(y_one_hot, y_score, average='micro')
    # 2、手动计算micro类型的AUC
    # 首先将矩阵y_one_hot和y_score展开，然后计算假正例率FPR和真正例率TPR
    fpr, tpr, thresholds = metrics.roc_curve(y_one_hot.ravel(), y_score.ravel())
    auc = metrics.auc(fpr, tpr)
    print
    '手动计算auc：', auc
    # 绘图
    mpl.rcParams['font.sans-serif'] = u'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    # FPR就是横坐标,TPR就是纵坐标
    plt.plot(fpr, tpr, c='r', lw=2, alpha=0.7, label=u'AUC=%.3f' % auc)
    plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
    plt.title(u'鸢尾花数据Logistic分类后的ROC和AUC', fontsize=17)
    plt.show()

import torch
import torch.nn as nn
import os
import numpy as np
from torchvision.datasets import ImageFolder
from utils.transform import get_transform_for_test
from senet.se_resnet import FineTuneSEResnet50
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

data_root = r'D:\TJU\GBDB\set113\set113_images\test1'  # 测试集路径
test_weights_path = r"C:\Users\admin\Desktop\fsdownload\epoch_0278_top1_70.565_'checkpoint.pth.tar'"  # 预训练模型参数
num_class = 113  # 类别数量
gpu = "cuda:0"


# mean=[0.948078, 0.93855226, 0.9332005], var=[0.14589554, 0.17054074, 0.18254866]
def test(model, test_path):
    # 加载测试集和预训练模型参数
    test_dir = os.path.join(data_root, 'test_images')
    class_list = list(os.listdir(test_dir))
    class_list.sort()
    transform_test = get_transform_for_test(mean=[0.948078, 0.93855226, 0.9332005],
                                            var=[0.14589554, 0.17054074, 0.18254866])
    test_dataset = ImageFolder(test_dir, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True, num_workers=1)
    checkpoint = torch.load(test_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    score_list = []  # 存储预测得分
    label_list = []  # 存储真实标签
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = model(inputs)
        # prob_tmp = torch.nn.Softmax(dim=1)(outputs) # (batchsize, nclass)
        score_tmp = outputs  # (batchsize, nclass)

        score_list.extend(score_tmp.detach().cpu().numpy())
        label_list.extend(labels.cpu().numpy())

    score_array = np.array(score_list)
    # 将label转换成onehot形式
    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], num_class)
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)

    print("score_array:", score_array.shape)  # (batchsize, classnum)
    print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum])

    # 调用sklearn库，计算每个类别对应的fpr和tpr
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    for i in range(num_class):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
    # micro
    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
    roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

    # macro
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_class)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])
    # Finally average it and compute AUC
    mean_tpr /= num_class
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])

    # 绘制所有类别平均的roc曲线
    plt.figure()
    lw = 2
    plt.plot(fpr_dict["micro"], tpr_dict["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr_dict["macro"], tpr_dict["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(num_class), colors):
        plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc_dict[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('set113_roc.jpg')
    plt.show()


if __name__ == '__main__':
    # 加载模型
    seresnet = FineTuneSEResnet50(num_class=num_class)
    device = torch.device(gpu)
    seresnet = seresnet.to(device)
    test(seresnet, test_weights_path)