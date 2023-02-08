# coding: utf-8
import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision.models as models
import pandas as pd

# 为了能读取到中间梯度定义的辅助函数
def extract(g):
    global features_grad
    features_grad = g


def draw_CAM(model, img_path,save_path, flag,transform=None, visual_heatmap=False):
    '''
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch 1
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # 文件夹
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(save_path)

    _,name = os.path.split(img_path)
    print(name)
    # 图像加载&预处理
    img = Image.open(img_path).convert('RGB')
    imgt = transform(img)
    imgt.unsqueeze_(0)
    imgt=imgt.to(device)
    # 获取模型输出的feature/score
    # 1.eval()
    # print(1)
    # exit()
    if flag == 'dense':
        features = model.features(imgt)
        x = model.avgpool(features)
        x = x.view(x.size(0),-1)
        output = model.classifier(x)

    else:
        x1=model.conv1(imgt)
        x2= model.bn1(x1)
        x3=model.relu(x2)
        x4= model.maxpool(x3)
        x5=model.layer1(x4)
        x6=model.layer2(x5)
        x7 = model.layer3(x6)
        features = model.layer4(x7)
        x = model.avgpool(features)
        x = x.view(x.size(0), -1)
        output = model.fc(x)

    # if flag =='res2':



    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output,1).item()
    pred_class = output[:, pred]

    # features.register_hook(extract)
    # pred_class.backward()  # 计算梯度
    features_grad = autograd.grad(pred_class, features, allow_unused=True)[0]

    grads = features_grad  # 获取梯度

    pooled_grads = F.adaptive_avg_pool2d(grads, (1,1))

    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    features = features[0]
    # 512是最后一层feature的通道数
    for i in range(features.shape[0]):
        features[i, ...] *= pooled_grads[i, ...]

    # 以下部分同Keras版实现
    heatmap = features.detach().cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()

    img = cv2.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
    cv2.imwrite(os.path.join(save_path,name),superimposed_img)  # 将图像保存到硬盘
    # print('ok')
    # exit()

img_size = 256
transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485],
                             std=[0.229])
    ])
# 1 = torch.load('../checkout/cut/densenet/06-12-17_220-1.000.pth')
# 1 = models.resnet50(pretrained = True)
# print(1)
# exit()
# 1.cuda()
# img_root = "E:\hzl\segmentation\\3d_seg\dataset\seg_data\\test\ILL\goufengbo\\No13_0602_0009_493310035_1.png"
# img_root = '../31_3.png'
model_r = torch.load('../checkout/cut/resnet/06-15-17_300-1.000.pth')
model_r2 = torch.load('../checkout/cut/res2net/06-14-11_300-1.000.pth')         #best06-14-09_165-0.860.pth
model_d = torch.load('../checkout/cut/densenet/06-15-14_220-1.000.pth')

# draw_CAM(1,img_root,transform = transform_val,)

# list1_dir = '../dataset/error'
# list2_dir = '../dataset/cut_error'
# imgers = os.listdir(list1_dir)
# imgcuters = os.listdir(list2_dir)
f = pd.read_csv('../list/test1.txt', header=None, sep=',')
dir1 = f[0]

path = '../cam'
if not os.path.exists(path):os.mkdir(path)

for img_path in dir1:
    # print(img_path)
    img_ori = img_path.replace('cut/','')
    # print(img_ori)
    # exit()

    model_f = 'densenet'
    draw_CAM(model_d, img_path, save_path=os.path.join(path, model_f, 'cut'), flag='dense', transform=transform_val)
    draw_CAM(model_d, img_ori, save_path=os.path.join(path, model_f, 'ori'), flag='dense', transform=transform_val)

    model_f = 'res2net'
    draw_CAM(model_r2, img_path, save_path=os.path.join(path, model_f, 'cut'), flag='res2', transform=transform_val)
    draw_CAM(model_r2, img_ori, save_path=os.path.join(path, model_f, 'ori'), flag='res2', transform=transform_val)

    model_f = 'resnet'
    draw_CAM(model_r, img_path,save_path=os.path.join(path, model_f, 'cut'),flag = 'res',transform=transform_val)
    draw_CAM(model_r, img_ori,save_path=os.path.join(path, model_f, 'ori'),flag = 'res',transform=transform_val)

# for img in imgcuters:
#     # img_p = os.path.abspath(img)
#     # print(img_p)
#     img_path = os.path.join(list2_dir,img)
#     print(img_path)
#     # exit()
#
#     model_f = 'densenet'
#     img_f = 'cut'
#     draw_CAM(model_d,img_path,save_path=os.path.join(path,model_f,img_f),flag='dense',transform=transform_val)
#     model_f = 'res2net'
#     draw_CAM(model_r2,img_path,save_path=os.path.join(path,model_f,img_f),flag = 'res2',transform=transform_val)
#     model_f = 'resnet'
#     draw_CAM(model_r, img_path, save_path=os.path.join(path, model_f, img_f), flag='res', transform=transform_val)
