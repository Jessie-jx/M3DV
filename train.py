import torch
torch.cuda.current_device()
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from MyDataSet import DealTrainset,DealDevset
from Model_Densenet import DenseNet
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 超参数设置
BATCH_SIZE = 16
NUM_EPOCHS = 50
LR = 0.0001
alpha = 1


# 定义训练批处理数据
dealTrainset = DealTrainset()
train_loader = DataLoader(dataset=dealTrainset,
                          batch_size=BATCH_SIZE,
                          shuffle=True
                          )

# 定义验证批处理数据
dealDevset = DealDevset()
dev_loader = DataLoader(dataset=dealDevset,
                        batch_size=BATCH_SIZE,
                        shuffle=False
                        )

# 定义损失函数和优化方式
net = DenseNet(drop_rate=0.5).to(device)
criterion = nn.CrossEntropyLoss()
# criterion = nn.BCELoss().cuda()
criterion = criterion.to(device)
optimizer = optim.Adam(net.parameters(), lr=LR)
# optimizer = optim.SGD(net.parameters(),lr=0.001, momentum=0.9)




# 保存模型的一些量
scores_perepoch = []
dev_auc = []
compare_dev_auc = 0
time = str(time.time())
save_path = 'parameter/' + time
isExist = os.path.exists(save_path)
if not isExist:
    os.makedirs(save_path)
    print('创建保存目录成功')
else:
    print('目录已存在')


# mixup
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


for epoch in range(NUM_EPOCHS):
    sum_loss = 0
    # 数据读取
    predict_value = torch.zeros(1)  # valiation predict:
    true_value = torch.zeros(1)  # true lable of valiation part:
    # 数据读取
    for i, data in enumerate(train_loader):
        Xtrain,Ytrain=data
        Xtrain = Xtrain.type(torch.FloatTensor)
        Ytrain = Ytrain.type(torch.LongTensor)
        Xtrain, Ytrain = Xtrain.to(device), Ytrain.to(device)
        Xtrain, targets_a, targets_b, lam = mixup_data(Xtrain, Ytrain,
                                                           1.)

        # 梯度清零
        optimizer.zero_grad()

        # 前后传播+后向传播
        outputs = net(Xtrain)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        # loss = criterion(outputs, Ytrain)
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
    print('第%d个epoch的loss: %.08f' % ((epoch + 1), sum_loss / 32))
    sum_loss = 0

    # 每跑完一次epoch计算验证集ROC
    with torch.no_grad():
        correct = 0
        total = 0
        save_score = np.zeros([1, 1])
        save_label = np.zeros([1, 1])
        for data in dev_loader:
            Xdev, Ydev = data
            Xdev = Xdev.type(torch.FloatTensor)
            Ydev = Ydev.type(torch.LongTensor)
            Xdev, Ydev = Xdev.to(device), Ydev.to(device)
            outputs = net(Xdev)

            dev_value = F.softmax(outputs, dim=1)
            dev_value_1 = dev_value[:, 1]
            dev_value_1 = dev_value_1.cpu()
            predict_value = torch.cat((predict_value, dev_value_1))

            dev_value_2 = Ydev
            dev_value_2 = dev_value_2.cpu().float()
            true_value = torch.cat((true_value, dev_value_2))

    predict_value = predict_value.detach().numpy()
    true_value = true_value.numpy()
    auc_score = roc_auc_score(true_value, predict_value)
    print('第%d个epoch的验证集AUROC:%.2f%%' % (epoch + 1, (100 * auc_score)))
    dev_auc.append(auc_score)
    scores_perepoch.append(save_score)
    if auc_score > compare_dev_auc:
        compare_dev_auc = auc_score
        file_path = save_path + '/' + str(auc_score) + '.pkl'
        torch.save(net.state_dict(), file_path)
        print('save model done!')

    predict_value = torch.zeros(1)
    true_value = torch.zeros(1)
    # 每跑完一次epoch计算训练集ROC
    with torch.no_grad():
        correct = 0
        total = 0
        for data in train_loader:
            X_train, Y_train = data
            X_train = X_train.type(torch.FloatTensor)
            Y_train = Y_train.type(torch.LongTensor)
            X_train, Y_train = X_train.to(device), Y_train.to(device)
            outputs = net(X_train)



            train_value = F.softmax(outputs, dim=1)
            train_value_1 = train_value[:, 1]
            train_value_1 = train_value_1.cpu()

            predict_value = torch.cat((predict_value, train_value_1))

            train_value_2 = Y_train
            train_value_2 = train_value_2.cpu().float()
            true_value = torch.cat((true_value, train_value_2))

    predict_value = predict_value.detach().numpy()
    true_value = true_value.numpy()
    auc_score = roc_auc_score(true_value, predict_value)
    print('第%d个epoch的训练集AUROC:%.2f%%' % (epoch + 1, (100 * auc_score)))

# 保存
torch.save(net.state_dict(), 'parameter/parameter0619_dense_1.pkl')
print('Save model done!')