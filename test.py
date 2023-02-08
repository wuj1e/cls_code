from __future__ import print_function, division

import os

import pandas as pd
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset import dataloader
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, classification_report, f1_score
import nmodel
from resnest.torch import resnest50
from sklearn.metrics import confusion_matrix
from metrics.ROC import draw_roc
from metrics.metircs import *
from scipy import stats

''''''

# ------------------------ 加载数据 --------------------------- #
# Data augmentation and normalization for training
# Just normalization for validation
# 定义预训练变换
# 数据预处理
img_size = 256
transform_val = transforms.Compose([
        # transforms.CenterCrop(img_size),
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
print(device)

# ------------------------ 载入模型并且训练 --------------------------- #



path = 'list/test.txt'
# f = pd.read_csv(path,header=None)
# list = open(path,'r').readlines()


f=open('testlabel.txt','w')


# 在test集上评估
def show_predictions(model, test_loaders):
    model.eval()
    images_handled = 0

    plt.figure()
    predall = []
    gtall = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loaders):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            # print(outputs)

            preds = torch.argmax(outputs, 1)
            preds = list(preds.detach().cpu().numpy())
            predall += preds
            gtall += list(labels.long()[:, 0].cpu().numpy())

            # f.write(imgpaths[0]+',标签：'+str(label[0])+',预测：'+str(pred[0])+'\n')

        tar_label = ['HEALTH','ILL']
        print(classification_report(gtall, predall, target_names=tar_label,digits=4))
        # draw_roc(predall,gtall,classnumber=4,title='Densenet121')
        # # # f1 = f1_score(gtall, predall, average='macro')
        acc_all = accuracy_score(gtall, predall)
        acc = list_(ACC(gtall,predall,2))

        pre = list_(precision(gtall,predall,2))
        # spe = list_(specificity(gtall,predall,2))
        # sen = list_(sensitivity(gtall,predall,2))
        # # # print(1)
        # print("label:{},pred:{}".format(gtall,predall))
        print("acc_all:%.4f"%acc_all)
        print("Accuracy:{},Precision:{}".format(acc,pre))
        plot_confusion_matrix(confusion_matrix(gtall,predall),classes=tar_label,title="Densenet121")
        # # return f1, acc
        return acc_all

testdata = dataloader(path, transform_val)
testloader = DataLoader(testdata, batch_size=8,shuffle=True,
                          num_workers=0, drop_last=False)  # define traindata shuffle参数隐藏说明是index是随机的

# model_path = r'checkout/cut/Resnest'
# models = os.listdir(model_path)
model = torch.load('checkout/cut/Dense/best07-18-09_34-0.945.pth')
show_predictions(model,testloader)

best_acc = 0.7
best=1
l=[]

# for pth in models:
#     # if '07-20' in pth:
#         model = torch.load(os.path.join(model_path,pth))
#         test_acc = show_predictions(model, testloader)
#         if test_acc >= best_acc:
#             best_acc = test_acc
#             best = pth
#         if test_acc>=0.8:
#             l.append(pth+str(test_acc))
# print("{}:acc{}".format(best,best_acc))
# print(l)


# show_predictions(1,testloader)