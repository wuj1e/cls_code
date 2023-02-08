# 导入库
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score
from dataset import dataloader
import torch.nn.parallel
from torch.utils.data import DataLoader
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
from nmodel.repvgg import *
from model import *
import torchsummaryX



def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    modellrnew = modellr * (0.1 ** (epoch // 10))
    print("lr:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew


# 定义训练过程
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pred_all = []
    gt_all = []
    gloss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        output = model(data)
        pred = torch.argmax(output, 1)
        pred = list(pred.detach().cpu().numpy())
        pred_all += pred
        gt_all += list(target.long()[:, 0].cpu().numpy())
        loss = criterion(output, target.squeeze().long())
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))
        gloss += loss.item()

    print(classification_report(gt_all, pred_all))
    f1 = f1_score(gt_all, pred_all, average='macro')
    acc = accuracy_score(gt_all, pred_all)
    print("f1:{},acc:{}".format(f1, acc))
    return gloss / len(train_loader), f1, acc
    # print(classification_report(gt_all, pred_all))


# 定义测试过程

def val(model, device, test_loader):
    model.eval()
    test_loss = 0
    pred_all = []
    gt_all = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).float().unsqueeze(1)
            output = model(data)
            pred = torch.argmax(output, 1)
            pred = list(pred.detach().cpu().numpy())
            pred_all += pred
            gt_all += list(target.long()[:, 0].cpu().numpy())
            test_loss += criterion(output, target.squeeze().long()).item()

    print(classification_report(gt_all, pred_all))
    f1 = f1_score(gt_all, pred_all)
    acc = accuracy_score(gt_all, pred_all)
    print(gt_all, pred_all)
    return f1, acc


if __name__ == '__main__':
    train_path = 'list/train.txt'
    val_path = 'list/val.txt'
    img_size = 256
    modellr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    print(device)
    # 1 = ResNet50(num_classes=2).to(device)

    mod = create_RepVGG_B1g2(deploy=False)
    model = repvgg_model_convert(mod, save_path='repvgg_deploy.pth').to(device)
    # input = torch.rand(4,3,256,256)
    # torchsummaryX.summary(model,input)
    # exit()

    # 1 = Densenet121(num_classes=2,pretrained=False)
    #
    # pre = 'checkout/cut/Dense/07-17-11_10-0.975.pth'
    # # 1.load_state_dict(torch.load(pre).state_dict())
    # model_dict = 1.state_dict()
    # pre_dict = torch.load(pre).state_dict()
    # # # 将pretrained_dict里不属于model_dict的键剔除掉
    # pre_dict = {k: v for k, v in pre_dict.items() if k in model_dict}
    # # 更新现有的model_dict
    # model_dict.update(pre_dict)
    # # # 加载我们真正需要的state_dict
    # 1.load_state_dict(model_dict)

    if torch.cuda.device_count() > 1:
        # device_ids = range(torch.cuda.device_count())
        device_ids = [0,1]
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.to(device)
    mod = 'Rep'
    print(model)


    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=modellr)
    optimizer  = optim.SGD(model.parameters(),lr=modellr,momentum=0.9,weight_decay=1e-4)
    BATCH_SIZE = 64
    EPOCHS = 150
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_data = dataloader(train_path, transform_train)
    val_data = dataloader(val_path, transform_val)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=True)  # define traindata shuffle参数隐藏说明是index是随机的
    test_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, drop_last=False)  # define traindata

    # 训练
    best_acc = 0.80

    loss_list = []
    f1_list = []
    acc_list = []

    for epoch in range(1, EPOCHS + 1):
        adjust_learning_rate(optimizer, epoch)
        train_loss,train_f1,train_acc = train(model, device, train_loader, optimizer, epoch)
        loss_list.append(train_loss)
        f1_list.append(train_f1)
        acc_list.append(train_acc)

        val_f1, val_acc = val(model, device, test_loader)
        print('f1 is {}, acc is {}'.format(val_f1, val_acc))
        # samdpath = 'checkout/cut/{mod}'
        # if not os.path.exists(samdpath):
        #     os.makedirs(samdpath)
        if  val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model,'./checkout/cut/{}/best{}_{}-{:.3f}.pth'.format(mod, time.strftime("%m-%d-%H", time.localtime()),
                                                                   epoch, best_acc))

        if val_acc > 0.70:
            if epoch % 10 == 0:
                torch.save(model,'./checkout/cut/{}/{}_{}-{:.3f}.pth'.format(mod, time.strftime("%m-%d-%H", time.localtime()),
                                                                   epoch, train_acc))


    x1 = range(0, EPOCHS)
    y1 = loss_list
    x2 = range(0, EPOCHS)
    y2 = f1_list
    x3 = range(0, EPOCHS)
    y3 = acc_list
    # 画图train_loss
    # plt.subplot(2, 1, 1)
    plt.plot(x1, y1)
    plt.xlabel('Epoches')
    plt.ylabel('Train loss')
    plt.savefig("Train_loss.jpg")
    plt.show()
    # plt.subplot(2, 1, 1)
    # plt.plot(x2, y2,'o-')
    # plt.xlabel('Epoches')
    # plt.ylabel('Train f1')

    # plt.subplot(2, 1, 2)
    # plt.subplot(2, 1, 1)
    plt.plot(x3, y3)
    # plt.title('Train Acc vs. Epoches')
    plt.xlabel('Epoches')
    plt.ylabel('Train Accuracy')
    plt.ylim(0, 1)
    plt.savefig('Train_acc.png')
    plt.show()