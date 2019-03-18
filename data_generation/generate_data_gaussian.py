import numpy as np
import torch
from trajectory_model.tracked_model import TrackedModel
from noiser.noiser import Noiser
import pickle

import matplotlib.pyplot as plt

clean_data = None
clean_label = None

mean = np.array((0.4914, 0.4822, 0.4465)).reshape(1,3,1)
var = np.array((0.2023, 0.1994, 0.2010)).reshape(1,3,1)

for i in range(1,6):
    with open('data/cifar-10/data_batch_' + str(i), 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
        if clean_data is None:
            temp = dict['data'].reshape(10000,3,-1).astype('float32')
            #print('temp is ', temp)
            clean_data = ((temp / 255 - mean) / var).reshape(10000,-1)
            #print('clean is ', clean_data)
            clean_label = np.array(dict['labels']).astype('int64').reshape(10000, 1)
        else:
            temp = dict['data'].reshape(10000, 3, -1).astype('float32')
            temp = ((temp / 255 - mean) / var).reshape(10000,-1)
            #print('temp is ' + temp)
            #print('clean is ' + clean_data)
            clean_data = np.vstack((clean_data, temp))
            clean_label = np.vstack((clean_label, np.array(dict['labels']).astype('int64').reshape(10000, 1)))

#clean = np.hstack((np.hstack((clean_data, clean_label)), np.arange(50000).reshape(50000,1)))
#index = np.zeros((clean.shape[0],1))
#np.savez('data/train_data/clean' ,data=clean, index=index)

#with open('data/cifar-10/test_batch', 'rb') as fo:
#    dict = pickle.load(fo, encoding='latin1')
#    temp = dict['data'].reshape(10000, 3, -1).astype('float32')
#    valid_data = ((temp / 255 - mean) / var).reshape(10000,-1)
#    valid_label = np.array(dict['labels']).astype('int64').reshape(10000, 1)

#valid = np.hstack((valid_data, valid_label))
#np.savez('data/valid_data/valid' ,data=valid)

noiser = Noiser()

for i in range(2, 5):
    data, index = noiser.generate(clean_data,clean_label.reshape(50000,), error_param={'sample_rate':(0.1)}, gaussian_param={'sample_rate':(i * 0.1),'mu':0,'var':0.5})
    np.savez('data/train_data/error_0.1_gaussian_0.' + str(i),data=data, index=index)