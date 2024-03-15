import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

import numpy as np
import matplotlib.pyplot as plt
import os

# https://blog.csdn.net/Gambler_Yushen/article/details/114855882?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522162692398016780357244827%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=162692398016780357244827&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_v2~rank_v29-6-114855882.first_rank_v2_pc_rank_v29&utm_term=%E5%9F%BA%E4%BA%8Eresnet%E7%9A%84%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB&spm=1018.2226.3001.4187
# 图像增强
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)), #随机裁剪到256*256
        transforms.RandomRotation(degrees=15),# 随机旋转
        transforms.RandomHorizontalFlip(p=0.5), # 依概率水平旋转
        transforms.CenterCrop(size=224),  # 中心裁剪到224*224符合resnet的输入要求
        transforms.ToTensor(),# 填充
        transforms.Normalize([0.485, 0.456, 0.406],#转化为tensor，并归一化至[0，-1]
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),#图像变换至256
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),#填充
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

#2加载数据集
dataset = 'dataset'
train_directory = os.path.join(dataset, 'train') #训练集的路径，os.path.join()函数是路径拼接函数
valid_directory = os.path.join(dataset, 'valid') #验证集的路径
test_directory = os.path.join(dataset , 'test')  #测试集路径


batch_size = 64 #分成32组
num_classes = 62 #图像的类，这里需要改，其余不变！！！

data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
     #imagefolder（root, transform），前者是图片路径，后者是对图片的变换，生成的数据类型是dataset
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
    'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['valid'])
}       #把dataset类型的数据放在数组里，便于通过键值调用

train_data_size = len(data['train'])#训练集的大小
valid_data_size = len(data['valid'])#验证集的大小
test_data_size = len(data['test'])#验证集的大小


train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
  #DataLoader(dataset, batch_size, shuffle) dataset数据类型；分组数；是否打乱
valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True)
test_data = DataLoader(data['test'], batch_size=batch_size, shuffle=True)

print("训练集数据量为：{}，验证集数据量为：{},测试集数据量为{}".format(train_data_size, valid_data_size,test_data_size))
  #分别打印出训练集和验证集的样本数量
print("2.加载数据完毕")


# 3加载模型，迁移学习
resnet50 = models.resnet50(pretrained=True) #开启预训练

for param in resnet50.parameters():#由于预训练的模型中的大多数参数已经训练好了，因此将requires_grad字段重置为false。
    param.requires_grad = False
    # 为了适应自己的数据集，将ResNet-50的最后一层替换为，
    # 将原来最后一个全连接层的输入喂给一个有256个输出单元的线性层，
    # 接着再连接ReLU层和Dropout层，然后是256 x 10的线性层，输出为5通道的softmax层。
fc_inputs = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Linear(fc_inputs, 1024),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(1024, 62),
    nn.LogSoftmax(dim=1)
)

loss_func = nn.NLLLoss()
optimizer = optim.Adam(resnet50.parameters())
print("3.模型载入完毕")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设备自行判断

def train_and_valid(model, loss_function, optimizer, epochs=25):

    history = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()  # 每轮开始时间记录
        print("Epoch: {}/{}".format(epoch + 1, epochs))  # 显示这是第多少轮

        model.train()  # 启用 Batch Normalization 和 Dropout。（随机去除神经元）

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data):  # 训练数据
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 因为这里梯度是累加的，所以每次记得清零
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():  # 用于通知dropout层和batchnorm层在train和val模式间切换。
            model.eval()  # model.eval()中的数据不会进行反向传播，但是仍然需要计算梯度；

            for j, (inputs, labels) in enumerate(valid_data):  # 验证数据
                inputs = inputs.to(device)  # 从valid_data里获得输入和标签
                labels = labels.to(device)

                outputs = model(inputs)  # 模型的输出

                loss = loss_function(outputs, labels)  # 损失计算

                valid_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)  # 在分类问题中,通常需要使用max()函数对tensor进行操作,求出预测值索引。
                # dim是max函数索引的维度0 / 1，0是每列的最大值，1是每行的最大值
                # 在多分类任务中我们并不需要知道各类别的预测概率，所以第一个tensor对分类任务没有帮助，而第二个tensor包含了最大概率的索引，所以在实际使用中我们仅获取第二个tensor即可。
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                valid_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1

        epoch_end = time.time()

        print(
            "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_valid_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start
            ))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'D:/compete_code/base_resnet50/model/' + '_model_' + str(epoch + 1) + '.pth')

    return model, history


def test(model, loss_function):

    resnet50.load_state_dict(torch.load('.\dataset' + '_model_' + '18' + '.pt'))
    test_loss = 0.0
    test_acc = 0.0
    test_start = time.time()
    with torch.no_grad():  # 用于通知dropout层和batchnorm层在train和val模式间切换。
        model.eval()  # model.eval()中的数据不会进行反向传播，但是仍然需要计算梯度；
    for j, (inputs, labels) in enumerate(test_data):  # 验证数据
        inputs = inputs.to(device)  # 从test_data里获得输入和标签
        labels = labels.to(device)
        outputs = model(inputs)  # 模型的输出
        loss = loss_function(outputs, labels)  # 损失计算
        test_loss += loss.item() * inputs.size(0)
        ret, predictions = torch.max(outputs.data, 1)  # 在分类问题中,通常需要使用max()函数对tensor进行操作,求出预测值索引。
        correct_counts = predictions.eq(labels.data.view_as(predictions))
        acc = torch.mean(correct_counts.type(torch.FloatTensor))
        test_acc += acc.item() * inputs.size(0)

    avg_test_loss = test_loss / test_data_size
    avg_test_acc = test_acc / test_data_size
    test_end = time.time()

    print(
        "test: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
            avg_test_loss, avg_test_acc * 100,
            test_end - test_start
        ))


istrain = 1
model = resnet50.to(device)
if istrain:
    num_epochs = 100
    trained_model, history = train_and_valid(resnet50, loss_func, optimizer, num_epochs)


ispicshow = 1
if ispicshow:
    history = np.array(history)
    plt.plot(history[:, 0:2])
    plt.legend(['Tr Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 4)
    plt.savefig(dataset + '_loss_curve.png')
    plt.show()

    plt.plot(history[:, 2:4])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig(dataset + '_accuracy_curve.png')
    plt.show()
