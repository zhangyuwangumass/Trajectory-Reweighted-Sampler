import numpy as np
import torch
import torch.optim as optim
from trajectory_model.tracked_model import TrackedModel
import torch.nn as nn
from model.resnet import ResNet18

import matplotlib.pyplot as plt


MAX_EPOCH = 150
MAX_ACCURACY = 1
BATCH_SIZE = 100
LR = 0.1
N = 50000

valid = np.load('data/valid_data/valid.npz')['data']
test = np.load('data/valid_data/test.npz')['data']
data = np.load('data/train_data/clean.npz')['data']
index = np.load('data/train_data/clean.npz')['index']

print('Stage 1 training starts!')

model = TrackedModel(ResNet18(), data, test)
train_his, val_his, class_his, train_recall, val_recall = model.train(
    optim.SGD(model.model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4), BATCH=BATCH_SIZE, MAX_EPOCH=MAX_EPOCH, MAX_ACCURACY=MAX_ACCURACY, USE_VAL=True)

N, EPOCH_ONE = train_his.shape
his = np.zeros_like(train_his)
for i in range(N):
    his[i] = train_his[np.where(class_his==i)]
np.savez('bin/record/baseline_cifar_test_first', his=his, index=index, train_recall=train_recall, val_recall=val_recall)
torch.save(model.model.state_dict(), 'bin/saved_model/params_resnet18_clean_first.pkl')

MAX_EPOCH = 50
MAX_ACCURACY = 1
LR = 0.01

print('Stage 2 training starts!')
model = TrackedModel(ResNet18(), data, test)
model.load_model('bin/saved_model/params_resnet18_clean_first.pkl')
train_his, val_his, class_his, train_recall, val_recall = model.train(
    optim.SGD(model.model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4), BATCH=BATCH_SIZE, MAX_EPOCH=MAX_EPOCH, MAX_ACCURACY=MAX_ACCURACY, USE_VAL=True)

N, EPOCH_TWO = train_his.shape
his = np.zeros_like(train_his)
for i in range(N):
    his[i] = train_his[np.where(class_his==i)]
np.savez('bin/record/baseline_cifar_test_second', his=his, index=index, train_recall=train_recall, val_recall=val_recall)
torch.save(model.model.state_dict(), 'bin/saved_model/params_resnet18_clean_second.pkl')

MAX_EPOCH = 55
MAX_ACCURACY = 1
LR = 0.001

print('Stage 3 training starts!')
model = TrackedModel(ResNet18(), data, test)
model.load_model('bin/saved_model/params_resnet18_clean_second.pkl')
train_his, val_his, class_his, train_recall, val_recall = model.train(
        optim.SGD(model.model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4), BATCH=BATCH_SIZE, MAX_EPOCH=MAX_EPOCH, MAX_ACCURACY=MAX_ACCURACY,
        USE_VAL=True)

N, EPOCH_ONE = train_his.shape
his = np.zeros_like(train_his)
for i in range(N):
    his[i] = train_his[np.where(class_his == i)]
np.savez('bin/record/baseline_cifar_test_third', his=his, index=index, train_recall=train_recall,
             val_recall=val_recall)
torch.save(model.model.state_dict(), 'bin/saved_model/params_resnet18_clean_third.pkl')