import numpy as np
import torch
import torch.optim as optim
from trajectory_model.tracked_model import TrackedModel
import torch.nn as nn
from model.resnet import ResNet18

import matplotlib.pyplot as plt


EPOCH = 10
BATCH_SIZE = 128
LR = 0.01
N = 50000

valid = np.load('data/valid_data/valid.npz')['data']
data = np.load('data/train_data/clean.npz')['data']
index = np.load('data/train_data/clean.npz')['index']

print('Epoch 10 training starts!')

model = TrackedModel(ResNet18(), data, valid)
train_his, val_his, class_his, train_recall, val_recall = model.train(
    optim.SGD(model.model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4), BATCH=BATCH_SIZE, MAX_EPOCH=EPOCH, USE_VAL=True)

his = np.zeros((N,EPOCH))
for i in range(N):
    his[i] = train_his[np.where(class_his==i)]
np.savez('bin/record/baseline_cifar_clean_first', his=his, index=index, train_recall=train_recall, val_recall=val_recall)
torch.save(model.model.state_dict(), 'bin/saved_model/params_resnet_clean_first.pkl')


for i in range(1, 5):
    print("Round " + str(i))
    data = np.load('data/train_data/error_0.' + str(i) + '.npz')['data']
    index = np.load('data/train_data/error_0.' + str(i) + '.npz')['index']

    model = TrackedModel(ResNet18(), data, valid)
    train_his, val_his, class_his, train_recall, val_recall = model.train(
        optim.SGD(model.model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4), BATCH=BATCH_SIZE, MAX_EPOCH=EPOCH, USE_VAL=True)
    his = np.zeros((N, EPOCH))
    for j in range(N):
        his[j] = train_his[np.where(class_his == j)]
    np.savez('bin/record/baseline_cifar_error_0.' + str(i) + '_first', his=his, index=index, train_recall=train_recall, val_recall=val_recall)
    torch.save(model.model.state_dict(), 'bin/saved_model/params_resnet_error_0.' + str(i) + '_first.pkl')
'''
LR = 0.01

print(data.shape)

model = TrackedModel(ResNet18(), data, valid)
model.load_model('bin/saved_model/params_resnet_clean_first.pkl')
train_his, val_his, class_his, train_recall, val_recall = model.train(
    optim.SGD(model.model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4), BATCH=BATCH_SIZE, MAX_EPOCH=EPOCH, USE_VAL=True)

his = np.zeros((N,EPOCH))
for i in range(N):
    his[i] = train_his[np.where(class_his==i)]
np.savez('bin/record/baseline_cifar_clean_second', his=his, index=index, train_recall=train_recall, val_recall=val_recall)
torch.save(model.model.state_dict(), 'bin/saved_model/params_resnet_clean_second.pkl')


for i in range(1, 5):
    print("Round " + str(i))
    data = np.load('data/train_data/error_0.' + str(i) + '_gaussian_0.1.npz')['data']
    index = np.load('data/train_data/error_0.' + str(i) + '_gaussian_0.1.npz')['index']

    model = TrackedModel(ResNet18(), data, valid)
    model.load_model('bin/saved_model/params_resnet_error_0.' + str(i) + '_gaussian_0.1_first.pkl')
    train_his, val_his, class_his, train_recall, val_recall = model.train(
        optim.SGD(model.model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4), BATCH=BATCH_SIZE, MAX_EPOCH=EPOCH, USE_VAL=True)
    his = np.zeros((N, EPOCH))
    for j in range(N):
        his[j] = train_his[np.where(class_his == j)]
    np.savez('bin/record/baseline_cifar_error_0.' + str(i) + '_gaussian_0.1_second', his=his, index=index, train_recall=train_recall, val_recall=val_recall)
    torch.save(model.model.state_dict(), 'bin/saved_model/params_resnet_error_0.' + str(i) + '_gaussian_0.1_second.pkl')

LR = 0.001

print(data.shape)

model = TrackedModel(ResNet18(), data, valid)
model.load_model('bin/saved_model/params_resnet_clean_second.pkl')
train_his, val_his, class_his, train_recall, val_recall = model.train(
        optim.SGD(model.model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4), BATCH=BATCH_SIZE, MAX_EPOCH=EPOCH,
        USE_VAL=True)

his = np.zeros((N, EPOCH))
for i in range(N):
    his[i] = train_his[np.where(class_his == i)]
np.savez('bin/record/baseline_cifar_clean_third', his=his, index=index, train_recall=train_recall,
             val_recall=val_recall)
torch.save(model.model.state_dict(), 'bin/saved_model/params_resnet_clean_third.pkl')


for i in range(1, 5):
    print("Round " + str(i))
    data = np.load('data/train_data/error_0.' + str(i) + '_gaussian_0.1.npz')['data']
    index = np.load('data/train_data/error_0.' + str(i) + '_gaussian_0.1.npz')['index']

    model = TrackedModel(ResNet18(), data, valid)
    model.load_model('bin/saved_model/params_resnet_error_0.' + str(i) + '_gaussian_0.1_second.pkl')
    train_his, val_his, class_his, train_recall, val_recall = model.train(
            optim.SGD(model.model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4), BATCH=BATCH_SIZE,
            MAX_EPOCH=EPOCH, USE_VAL=True)
    his = np.zeros((N, EPOCH))
    for j in range(N):
        his[j] = train_his[np.where(class_his == j)]
    np.savez('bin/record/baseline_cifar_error_0.' + str(i) + '_gaussian_0.1_third', his=his, index=index,
                 train_recall=train_recall, val_recall=val_recall)
    torch.save(model.model.state_dict(),
                   'bin/saved_model/params_resnet_error_0.' + str(i) + '_gaussian_0.1_third.pkl')


#################
print('Epoch 20 training starts!')

EPOCH = 20
BATCH_SIZE = 128
LR = 0.1
N = 50000

valid = np.load('data/valid_data/valid.npz')['data']
data = np.load('data/train_data/clean.npz')['data']
index = np.load('data/train_data/clean.npz')['index']
print(data.shape)

model = TrackedModel(ResNet18(), data, valid)
train_his, val_his, class_his, train_recall, val_recall = model.train(
    optim.SGD(model.model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4), BATCH=BATCH_SIZE, MAX_EPOCH=EPOCH, USE_VAL=True)

his = np.zeros((N,EPOCH))
for i in range(N):
    his[i] = train_his[np.where(class_his==i)]
np.savez('bin/record/baseline_cifar_clean_first_20', his=his, index=index, train_recall=train_recall, val_recall=val_recall)
torch.save(model.model.state_dict(), 'bin/saved_model/params_resnet_clean_first_20.pkl')


for i in range(1, 5):
    print("Round " + str(i))
    data = np.load('data/train_data/error_0.' + str(i) + '_gaussian_0.1.npz')['data']
    index = np.load('data/train_data/error_0.' + str(i) + '_gaussian_0.1.npz')['index']

    model = TrackedModel(ResNet18(), data, valid)
    train_his, val_his, class_his, train_recall, val_recall = model.train(
        optim.SGD(model.model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4), BATCH=BATCH_SIZE, MAX_EPOCH=EPOCH, USE_VAL=True)
    his = np.zeros((N, EPOCH))
    for j in range(N):
        his[j] = train_his[np.where(class_his == j)]
    np.savez('bin/record/baseline_cifar_error_0.' + str(i) + '_gaussian_0.1_first_20', his=his, index=index, train_recall=train_recall, val_recall=val_recall)
    torch.save(model.model.state_dict(), 'bin/saved_model/params_resnet_error_0.' + str(i) + '_gaussian_0.1_first_20.pkl')

LR = 0.01

print(data.shape)

model = TrackedModel(ResNet18(), data, valid)
model.load_model('bin/saved_model/params_resnet_clean_first_20.pkl')
train_his, val_his, class_his, train_recall, val_recall = model.train(
    optim.SGD(model.model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4), BATCH=BATCH_SIZE, MAX_EPOCH=EPOCH, USE_VAL=True)

his = np.zeros((N,EPOCH))
for i in range(N):
    his[i] = train_his[np.where(class_his==i)]
np.savez('bin/record/baseline_cifar_clean_second_20', his=his, index=index, train_recall=train_recall, val_recall=val_recall)
torch.save(model.model.state_dict(), 'bin/saved_model/params_resnet_clean_second_20.pkl')


for i in range(1, 5):
    print("Round " + str(i))
    data = np.load('data/train_data/error_0.' + str(i) + '_gaussian_0.1.npz')['data']
    index = np.load('data/train_data/error_0.' + str(i) + '_gaussian_0.1.npz')['index']

    model = TrackedModel(ResNet18(), data, valid)
    model.load_model('bin/saved_model/params_resnet_error_0.' + str(i) + '_gaussian_0.1_first_20.pkl')
    train_his, val_his, class_his, train_recall, val_recall = model.train(
        optim.SGD(model.model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4), BATCH=BATCH_SIZE, MAX_EPOCH=EPOCH, USE_VAL=True)
    his = np.zeros((N, EPOCH))
    for j in range(N):
        his[j] = train_his[np.where(class_his == j)]
    np.savez('bin/record/baseline_cifar_error_0.' + str(i) + '_gaussian_0.1_second_20', his=his, index=index, train_recall=train_recall, val_recall=val_recall)
    torch.save(model.model.state_dict(), 'bin/saved_model/params_resnet_error_0.' + str(i) + '_gaussian_0.1_second_20.pkl')

LR = 0.001

print(data.shape)

model = TrackedModel(ResNet18(), data, valid)
model.load_model('bin/saved_model/params_resnet_clean_second_20.pkl')
train_his, val_his, class_his, train_recall, val_recall = model.train(
        optim.SGD(model.model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4), BATCH=BATCH_SIZE, MAX_EPOCH=EPOCH,
        USE_VAL=True)

his = np.zeros((N, EPOCH))
for i in range(N):
    his[i] = train_his[np.where(class_his == i)]
np.savez('bin/record/baseline_cifar_clean_third_20', his=his, index=index, train_recall=train_recall,
             val_recall=val_recall)
torch.save(model.model.state_dict(), 'bin/saved_model/params_resnet_clean_third_20.pkl')


for i in range(1, 5):
    print("Round " + str(i))
    data = np.load('data/train_data/error_0.' + str(i) + '_gaussian_0.1.npz')['data']
    index = np.load('data/train_data/error_0.' + str(i) + '_gaussian_0.1.npz')['index']

    model = TrackedModel(ResNet18(), data, valid)
    model.load_model('bin/saved_model/params_resnet_error_0.' + str(i) + '_gaussian_0.1_second_20.pkl')
    train_his, val_his, class_his, train_recall, val_recall = model.train(
            optim.SGD(model.model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4), BATCH=BATCH_SIZE,
            MAX_EPOCH=EPOCH, USE_VAL=True)
    his = np.zeros((N, EPOCH))
    for j in range(N):
        his[j] = train_his[np.where(class_his == j)]
    np.savez('bin/record/baseline_cifar_error_0.' + str(i) + '_gaussian_0.1_third_20', his=his, index=index,
                 train_recall=train_recall, val_recall=val_recall)
    torch.save(model.model.state_dict(),
                   'bin/saved_model/params_resnet_error_0.' + str(i) + '_gaussian_0.1_third_20.pkl')
'''