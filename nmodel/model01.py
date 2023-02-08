import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models


class MobileNet(nn.Module):
    def __init__(self, num_classes=2):   # num_classes，此处为 二分类值为2
        super(MobileNet, self).__init__()
        net = models.mobilenet_v2(pretrained=True)   # 从预训练模型加载VGG16网络参数
        net.con
        net.classifier = nn.Sequential()  # 将分类层置空，下面将改变我们的分类层
        self.features = net  # 保留VGG16的特征层
        self.classifier = nn.Sequential(    # 定义自己的分类层
                nn.Linear(1280, 1000),  #512 * 7 * 7不能改变 ，由VGG16网络决定的，第二个参数为神经元个数可以微调
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, num_classes=2):   # num_classes，此处为 二分类值为2
        super(ResNet50, self).__init__()
        net = models.resnet50(pretrained=True)   # 从预训练模型加载VGG16网络参数
        net.conv1 = nn.Conv2d(16*3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        net.classifier = nn.Sequential()  # 将分类层置空，下面将改变我们的分类层
        self.features = net  # 保留VGG16的特征层
        self.classifier = nn.Sequential(    # 定义自己的分类层
                nn.Linear(1000, 100),  #1000不能改变 ，由VGG16网络决定的，第二个参数为神经元个数可以微调
                nn.Softmax(),
                nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Inception(nn.Module):
    def __init__(self, num_classes=2):   # num_classes，此处为 二分类值为2
        super(Inception, self).__init__()
        net = models.inception_v3(pretrained=True)   # 从预训练模型加载VGG16网络参数
        net.classifier = nn.Sequential()  # 将分类层置空，下面将改变我们的分类层
        self.features = net  # 保留VGG16的特征层
        self.classifier = nn.Sequential(    # 定义自己的分类层
                nn.Linear(1000, 1000),  #1000不能改变 ，由VGG16网络决定的，第二个参数为神经元个数可以微调
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.max_pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.max_pool3 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(64, 128, 3)
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.max_pool4 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(4608, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        in_size = x.size(0)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max_pool3(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.max_pool4(x)
        # 展开
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x