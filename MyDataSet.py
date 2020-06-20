# Data Loader
from torch.utils.data import Dataset
import random
import torch
import os
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import time
from DataTransform import Rotate1,Rotate2,Flip


data_dir = 'e:/data/'
patients_train = os.listdir(data_dir + 'train_val/train_val/')
patients_train.sort(key=lambda x: int(x[9:-4]))
patients_test = os.listdir(data_dir + 'test/test/')
patients_test.sort(key=lambda x: int(x[9:-4]))
labels_df = pd.read_csv('e:/data/train_val.csv', index_col=0)

data_origin = []
data_test = []
labels = labels_df['label'].values
names_train = []
names_test = []

x, y, z = 32, 32, 32
x_begin = int((100 - x) / 2)
x_end = x_begin + x
y_begin = int((100 - y) / 2)
y_end = y_begin + y
z_begin = int((100 - z) / 2)
z_end = z_begin + z

for num_train, patient_train in enumerate(patients_train):
    patient_train_name = patient_train[0:-4]
    names_train.append(patient_train_name)
    path_train = data_dir + 'train_val/train_val/' + patient_train
    img_data_train = np.load(path_train)
    voxel_train = img_data_train['voxel'].astype(np.int32)
    voxel_train_crop = voxel_train[x_begin:x_end, y_begin:y_end, z_begin:z_end]
    data_origin.append(voxel_train_crop)

for num_test, patient_test in enumerate(patients_test):
    names_test.append(patient_test[0:-4])
    path_test = data_dir + 'test/test/' + patient_test
    img_data_test = np.load(path_test)
    voxel_test = img_data_test['voxel'].astype(np.int32)
    voxel_test_crop = voxel_test[x_begin:x_end, y_begin:y_end, z_begin:z_end]
    data_test.append(voxel_test_crop)

data_origin = np.reshape(data_origin, [465, 32 * 32 * 32])
data_test = np.reshape(data_test, [117, 32 * 32 * 32])

ss = StandardScaler()
data_origin = ss.fit_transform(data_origin)
data_test = ss.transform(data_test)

data_origin = np.reshape(data_origin, [465, 1, 32, 32, 32])
data_test = np.reshape(data_test, [117, 1, 32, 32, 32])

# 划分验证集和训练集
kf = KFold(n_splits=5, shuffle=True, random_state=int(time.time()))
for train_index, dev_index in kf.split(data_origin):
    data_train, data_dev = np.array(data_origin)[train_index], np.array(data_origin)[dev_index]
    labels_train, labels_dev = np.array(labels)[train_index], np.array(labels)[dev_index]


class DealTrainset(Dataset):
    def __init__(self):
        global data_train,labels_train
        # data_train, labels_train = Rotate1(data_train,labels_train,1.5)
        #data_train, labels_train = Flip(data_train, labels_train, 1.3)
        # data_train, labels_train = Rotate2data_train, labels_train, 1.5)
        self.x_data = torch.from_numpy(data_train)
        self.y_data = torch.from_numpy(labels_train)
        self.len = labels_train.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class DealDevset(Dataset):
    def __init__(self):
        self.x_data = torch.from_numpy(data_dev)
        self.y_data = torch.from_numpy(labels_dev)
        self.len = labels_dev.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class DealTestset(Dataset):
    def __init__(self):
        self.x_data = torch.from_numpy(data_test)
        self.len = data_test.shape[0]

    def __getitem__(self, index):
        return self.x_data[index]

    def __len__(self):
        return self.len