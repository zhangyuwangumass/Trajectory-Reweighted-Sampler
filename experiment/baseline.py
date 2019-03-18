import numpy as np
import torch
import torch.optim as optim
from trajectory_model.tracked_model import TrackedModel
from noiser.noiser import Noiser
from model.linear import LinearModel
from model.conv import ConvModel, DeeperConvModel, ShallowerConvModel

import matplotlib.pyplot as plt


LR = 0.001

clean_data = np.load('data/dataA.npz')['train_image']
clean_label = np.load('data/dataA.npz')['train_label']
N = 6000
VAL = 600

valid_data = clean_data[N-VAL:N]
valid_label = clean_label[N-VAL:N]
valid = np.hstack((valid_data, valid_label.reshape(VAL,1)))


clean_data = clean_data[:N-VAL]
clean_label = clean_label[:N-VAL]
clean = np.hstack((clean_data, clean_label.reshape(N-VAL,1)))
clean = np.hstack((clean, np.zeros((N-VAL,1))))

N -= VAL
EPOCH = 30

noiser = Noiser()



data = noiser.generate(clean_data,clean_label, error_param={'sample_rate':0.8}, gaussian_param={'sample_rate':0.1,'mu':0,'var':0.5})
model = TrackedModel(ConvModel(), data, valid)

train_his, val_his, class_his, train_recall, val_recall = model.train(optim.Adam(model.model.parameters(), lr=0.0001, betas=(0.5, 0.999)), MAX_EPOCH=EPOCH, USE_VAL=True)

#plt.plot(np.arange(EPOCH), train_recall, label="train recall with 90% noise")
plt.plot(np.arange(EPOCH), val_recall, label="val recall with 90% noise")

np.savez('bin/record/baseline_0.9noise', train_recall=train_recall, val_recall=val_recall)

###################

data = noiser.generate(clean_data,clean_label, error_param={'sample_rate':0.7}, gaussian_param={'sample_rate':0.1,'mu':0,'var':0.5})
model = TrackedModel(ConvModel(), data, valid)
train_his, val_his, class_his, train_recall, val_recall = model.train(optim.Adam(model.model.parameters(), lr=0.0001, betas=(0.5, 0.999)), MAX_EPOCH=EPOCH, USE_VAL=True)

np.savez('bin/record/baseline_0.8noise', train_recall=train_recall, val_recall=val_recall)

#plt.plot(np.arange(EPOCH), train_recall, label="train recall with 15% noise")
plt.plot(np.arange(EPOCH), val_recall, label="val recall with 80% noise")

###################

data = noiser.generate(clean_data,clean_label, error_param={'sample_rate':0.6}, gaussian_param={'sample_rate':0.1,'mu':0,'var':0.5})
model = TrackedModel(ConvModel(), data, valid)
train_his, val_his, class_his, train_recall, val_recall = model.train(optim.Adam(model.model.parameters(), lr=0.0001, betas=(0.5, 0.999)), MAX_EPOCH=EPOCH, USE_VAL=True)

np.savez('bin/record/baseline_0.7noise', train_recall=train_recall, val_recall=val_recall)

#plt.plot(np.arange(EPOCH), train_recall, label="train recall with 15% noise")
plt.plot(np.arange(EPOCH), val_recall, label="val recall with 70% noise")
###################

data = noiser.generate(clean_data,clean_label, error_param={'sample_rate':0.5}, gaussian_param={'sample_rate':0.1,'mu':0,'var':0.5})
model = TrackedModel(ConvModel(), data, valid)
train_his, val_his, class_his, train_recall, val_recall = model.train(optim.Adam(model.model.parameters(), lr=0.0001, betas=(0.5, 0.999)), MAX_EPOCH=EPOCH, USE_VAL=True)

np.savez('bin/record/baseline_0.6noise', train_recall=train_recall, val_recall=val_recall)

#plt.plot(np.arange(EPOCH), train_recall, label="train recall with 15% noise")
plt.plot(np.arange(EPOCH), val_recall, label="val recall with 60% noise")

###################

data = noiser.generate(clean_data,clean_label, error_param={'sample_rate':0.4}, gaussian_param={'sample_rate':0.1,'mu':0,'var':0.5})
model = TrackedModel(ConvModel(), data, valid)
train_his, val_his, class_his, train_recall, val_recall = model.train(optim.Adam(model.model.parameters(), lr=LR, betas=(0.5, 0.999)), MAX_EPOCH=EPOCH, USE_VAL=True)

np.savez('bin/record/baseline_0.5noise', train_recall=train_recall, val_recall=val_recall)

#plt.plot(np.arange(EPOCH), train_recall, label="train recall with 50% noise")
plt.plot(np.arange(EPOCH), val_recall, label="val recall with 50% noise")

####################

data = noiser.generate(clean_data,clean_label, error_param={'sample_rate':0.3}, gaussian_param={'sample_rate':0.1,'mu':0,'var':0.5})
model = TrackedModel(ConvModel(), data, valid)
train_his, val_his, class_his, train_recall, val_recall = model.train(optim.Adam(model.model.parameters(), lr=0.0001, betas=(0.5, 0.999)), MAX_EPOCH=EPOCH, USE_VAL=True)

np.savez('bin/record/baseline_0.4noise', train_recall=train_recall, val_recall=val_recall)

#plt.plot(np.arange(EPOCH), train_recall, label="train recall with 20% noise")
plt.plot(np.arange(EPOCH), val_recall, label="val recall with 40% noise")

####################

data = noiser.generate(clean_data,clean_label, error_param={'sample_rate':0.2}, gaussian_param={'sample_rate':0.1,'mu':0,'var':0.5})
model = TrackedModel(ConvModel(), data, valid)
train_his, val_his, class_his, train_recall, val_recall = model.train(optim.Adam(model.model.parameters(), lr=0.0001, betas=(0.5, 0.999)), MAX_EPOCH=EPOCH, USE_VAL=True)

np.savez('bin/record/baseline_0.3noise', train_recall=train_recall, val_recall=val_recall)

#plt.plot(np.arange(EPOCH), train_recall, label="train recall with 20% noise")
plt.plot(np.arange(EPOCH), val_recall, label="val recall with 30% noise")

####################

data = noiser.generate(clean_data,clean_label, error_param={'sample_rate':0.1}, gaussian_param={'sample_rate':0.1,'mu':0,'var':0.5})
model = TrackedModel(ConvModel(), data, valid)
train_his, val_his, class_his, train_recall, val_recall = model.train(optim.Adam(model.model.parameters(), lr=0.0001, betas=(0.5, 0.999)), MAX_EPOCH=EPOCH, USE_VAL=True)

np.savez('bin/record/baseline_0.2noise', train_recall=train_recall, val_recall=val_recall)

#plt.plot(np.arange(EPOCH), train_recall, label="train recall with 20% noise")
plt.plot(np.arange(EPOCH), val_recall, label="val recall with 20% noise")

'''
data = noiser.generate(clean_data,clean_label, error_param={'sample_rate':0.2}, gaussian_param={'sample_rate':0.1,'mu':0,'var':0.5})
model = TrackedModel(ConvModel(), data, valid)
train_his, val_his, class_his, train_recall, val_recall = model.train(optim.Adam(model.model.parameters(), lr=LR, betas=(0.5, 0.999)), MAX_EPOCH=EPOCH, USE_VAL=True)

np.savez('bin/record/baseline_0.3noise', train_recall=train_recall, val_recall=val_recall)

#plt.plot(np.arange(EPOCH), train_recall, label="train recall with 20% noise")
plt.plot(np.arange(EPOCH), val_recall, label="val recall with 30% noise")


model = TrackedModel(ConvModel(), clean, valid)
train_his, val_his, class_his, train_recall, val_recall = model.train(optim.Adam(model.model.parameters(), lr=LR, betas=(0.5, 0.999)), MAX_EPOCH=EPOCH, USE_VAL=True)

np.savez('bin/record/baseline_clean', train_recall=train_recall, val_recall=val_recall)

#plt.plot(np.arange(EPOCH), train_recall, label="train recall with 20% noise")
plt.plot(np.arange(EPOCH), val_recall, label="val recall with 0% noise")
'''
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')

plt.show()