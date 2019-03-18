import numpy as np
import matplotlib.pyplot as plt

weighted_train_recall = np.load('bin/record/weighted_train_grad.npz')['train_recall']
weighted_val_recall = np.load('bin/record/weighted_train_grad.npz')['val_recall']

unweighted_train_recall = np.load('bin/record/unweighted_train_shallow.npz')['train_recall']
unweighted_val_recall = np.load('bin/record/unweighted_train_shallow.npz')['val_recall']


ROUND, EPOCH = weighted_train_recall.shape

plt.subplot(1,2,1)
#plt.plot(np.arange(EPOCH), train_loss, label='train loss')
#plt.plot(np.arange(EPOCH), val_loss, label='val loss')


plt.plot(np.arange(EPOCH), unweighted_train_recall.sum(axis=0), color='b', label='train recall without weight')
plt.plot(np.arange(EPOCH), weighted_train_recall.sum(axis=0), color='r', label='train recall with weight')

plt.legend()

plt.subplot(1,2,2)
#plt.plot(np.arange(EPOCH), train_recall, label='train recall')
#plt.plot(np.arange(EPOCH), val_recall, label='val recall')

plt.plot(np.arange(EPOCH), unweighted_val_recall.sum(axis=0), color='b', label='validation recall without weight')
plt.plot(np.arange(EPOCH), weighted_val_recall.sum(axis=0), color='r', label='validation recall with weight')

plt.legend()
plt.show()

