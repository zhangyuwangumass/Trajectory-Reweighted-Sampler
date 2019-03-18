import numpy as np
import torch
import torch.optim as optim
from trajectory_model.weighted_model import WeightedModel
from model.resnet import ResNet18
from sampler.sampler import Sampler
import random
import scipy.stats as st

COMP = 6

train_data = np.load('data/train_data/error_0.4_gaussian_0.1.npz')['data']
val_data = np.load('data/valid_data/valid.npz')['data']

index = np.load('bin/record/stats_baseline_cifar_error_0.4_gaussian_0.1_first.npz')['pred']

resnet_model = ResNet18().to(device=torch.device('cuda'))
#resnet_model.load_state_dict(torch.load('bin/saved_model/params_resnet_error_0.4_gaussian_0.1_first.pkl'))

weighted_sampler = Sampler(index, COMP)

model = WeightedModel(COMP, resnet_model, weighted_sampler, train_data, val_data)
prev_state = model.model.state_dict()

NUM = 1
EPOCH = 30
LR = 0.01


train_loss_history, train_recall_history, val_loss_history, val_recall_history = \
    model.train(optim.SGD(model.model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4), 30000, MODE='none', MAX_EPOCH=EPOCH)
np.savez('bin/record/from_clean_0.01/unweighted_resnet_error_0.4_gaussian_0.1_first',train_loss=train_loss_history, train_recall=train_recall_history, val_loss=val_loss_history, val_recall=val_recall_history)


model.model.load_state_dict(prev_state)
train_loss_history, train_recall_history, val_loss_history, val_recall_history = \
    model.train(optim.SGD(model.model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4), 30000, MODE='naive', MAX_EPOCH=EPOCH)
np.savez('bin/record/from_clean_0.01/weighted_naive_resnet_error_0.4_gaussian_0.1_first',train_loss=train_loss_history, train_recall=train_recall_history, val_loss=val_loss_history, val_recall=val_recall_history)

model.model.load_state_dict(prev_state)
#model.reset('bin/saved_model/params_resnet_error_0.4_gaussian_0.1_first.pkl')
train_loss_history, train_recall_history, val_loss_history, val_recall_history = model.train(
    optim.SGD(model.model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4), 30000, MODE='conf', MAX_EPOCH=EPOCH)
np.savez('bin/record/from_clean_0.01/weighted_conf_resnet_error_0.4_gaussian_0.1_first',train_loss=train_loss_history, train_recall=train_recall_history, val_loss=val_loss_history, val_recall=val_recall_history)


#model.reset('bin/saved_model/params_resnet_error_0.4_gaussian_0.1_first.pkl')
#train_loss_history, train_recall_history, val_loss_history, val_recall_history = model.train(
#    optim.SGD(model.model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4), 30000, MODE='shifted_conf', MAX_EPOCH=EPOCH)
#np.savez('bin/record/lr0.1/weighted_shifted_conf_resnet_error_0.4_gaussian_0.1_first',train_loss=train_loss_history, train_recall=train_recall_history, val_loss=val_loss_history, val_recall=val_recall_history)

model.model.load_state_dict(prev_state)
#model.reset('bin/saved_model/params_resnet_error_0.4_gaussian_0.1_first.pkl')
train_loss_history, train_recall_history, val_loss_history, val_recall_history = model.train(
    optim.SGD(model.model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4), 30000, MODE='grad', MAX_EPOCH=EPOCH)
np.savez('bin/record/from_clean_0.01/weighted_grad_resnet_error_0.4_gaussian_0.1_first',train_loss=train_loss_history, train_recall=train_recall_history, val_loss=val_loss_history, val_recall=val_recall_history)

model.model.load_state_dict(prev_state)
#model.reset('bin/saved_model/params_resnet_error_0.4_gaussian_0.1_first.pkl')
train_loss_history, train_recall_history, val_loss_history, val_recall_history = model.train(
    optim.SGD(model.model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4), 30000, MODE='valid_loss', MAX_EPOCH=EPOCH)
np.savez('bin/record/from_clean_0.01/weighted_valid_loss_resnet_error_0.4_gaussian_0.1_first',train_loss=train_loss_history, train_recall=train_recall_history, val_loss=val_loss_history, val_recall=val_recall_history)
