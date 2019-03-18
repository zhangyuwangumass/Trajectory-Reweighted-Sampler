import numpy as np
import random
import torch
from trajectory_model.tracked_model import TrackedModel
from noiser.noiser import Noiser
from utils.transformer import Transformer
import pickle

import matplotlib.pyplot as plt
from PIL import Image

transformer = Transformer()

clean_data = None
clean_label = None

mean = np.array((0.4914, 0.4822, 0.4465)).reshape(1,3,1)
var = np.array((0.2023, 0.1994, 0.2010)).reshape(1,3,1)

for i in range(1,6):
    with open('data/cifar-10/data_batch_' + str(i), 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
        if clean_data is None:
            temp = dict['data'].reshape(10000,3,32,32).astype('float32')
            #print('temp is ', temp)
            for j in range(temp.shape[0]):
                img = temp[j]
                img = transformer.random_crop(img, (32, 32), padding=4)
                img = transformer.random_horizontal_flip(img)
                print(img.shape)

                imgs = img.reshape(3,-1)
                for j in range(3):
                    image = imgs[j].reshape(32, 32)
                    plt.figure("Image")
                    plt.imshow(image)
                    plt.axis('on')
                    plt.title('image')
                    plt.show()
                temp[j] = img
            temp = temp.reshape(10000,3,-1)
            clean_data = ((temp / 255 - mean) / var).reshape(10000,-1)
            #print('clean is ', clean_data)
            clean_label = np.array(dict['labels']).astype('int64').reshape(10000, 1)
        else:
            temp = dict['data'].reshape(10000, 3, 32, 32).astype('float32')
            for j in range(temp.shape[0]):
                img = temp[j]
                img = transformer.random_crop(img, (32, 32), padding=4)
                img = transformer.random_horizontal_flip(img)
                temp[j] = img
            temp = temp.reshape(10000, 3, -1)
            temp = ((temp / 255 - mean) / var).reshape(10000,-1)
            clean_data = np.vstack((clean_data, temp))
            clean_label = np.vstack((clean_label, np.array(dict['labels']).astype('int64').reshape(10000, 1)))

clean = np.hstack((np.hstack((clean_data, clean_label)), np.arange(50000).reshape(50000,1)))
index = np.zeros((clean.shape[0],1))
np.savez('data/train_data/clean' ,data=clean, index=index)

with open('data/cifar-10/test_batch', 'rb') as fo:
    dict = pickle.load(fo, encoding='latin1')
    temp = dict['data'].reshape(10000, 3, -1).astype('float32')
    valid_data = ((temp / 255 - mean) / var).reshape(10000,-1)
    valid_label = np.array(dict['labels']).astype('int64').reshape(10000, 1)

valid_size = 1000

real = np.hstack((valid_data, valid_label))
valid_index = random.sample(range(10000), valid_size)
valid = real[valid_index]
#test = np.delete(real, valid_index, axis=0)
test = real

np.savez('data/valid_data/valid' ,data=valid)
np.savez('data/valid_data/test', data=test)


'''
noiser = Noiser()

for i in range(1, 5):
    data, index = noiser.generate(clean_data,clean_label.reshape(50000,), error_param={'sample_rate':(i * 0.1)}, gaussian_param={'sample_rate':0.1,'mu':0,'var':0.5})
    np.savez('data/train_data/error_0.' + str(i) + '_gaussian_0.1',data=data, index=index)
'''