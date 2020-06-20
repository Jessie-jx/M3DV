import torch.nn.functional as F
from torch.utils.data import DataLoader
from MyDataSet import DealTestset
from Model_Densenet import DenseNet
import torch
torch.cuda.current_device()
import numpy as np
from MyDataSet import names_test



# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dealTestset = DealTestset()
test_loader = DataLoader(dataset=dealTestset,
                         batch_size=8,
                         shuffle=False
                         )
torch.manual_seed(11)
save_score1 = np.zeros((1, 1))
predict_value = []
net = DenseNet(drop_rate=0.5).to(device)

net.load_state_dict(torch.load('/best paras.pkl'))
with torch.no_grad():
    for data in test_loader:
        Xtest = data
        Xtest = Xtest.type(torch.FloatTensor)
        Xtest = Xtest.to(device)
        outputs = net(Xtest)

        # 预测

        test_value = F.softmax(outputs, dim=1)
        test_value_1 = test_value[:, 1]
        test_value_1 = test_value_1.cpu()
        predict_value.extend(test_value_1.detach().numpy().tolist())

import csv
#stag_01_submit
csvFile = open("/test_submit.csv", "w",newline='')            #创建csv文件
writer = csv.writer(csvFile)                  #创建写的对象
#先写入columns_name
writer.writerow(["name","predicted"])     #写入列的名称

for i in range(len(names_test)):
    writer.writerow([names_test[i],predict_value[i]])

csvFile.close()

