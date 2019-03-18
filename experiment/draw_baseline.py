import numpy as np
import matplotlib.pyplot as plt


recall1 = np.load('bin/record/baseline_cifar_wrn28_first.npz')['val_recall']
recall2 = np.load('bin/record/baseline_cifar_wrn28_second.npz')['val_recall']
recall3 = np.load('bin/record/baseline_cifar_wrn28_third.npz')['val_recall']

EPOCH = recall1.shape[0] + recall2.shape[0] + recall3.shape[0]

plt.plot(np.arange(EPOCH), np.concatenate((recall1,recall2,recall3)), label='test recall')
plt.legend()

train1 = np.load('bin/record/baseline_cifar_wrn28_first.npz')['train_recall']
train2 = np.load('bin/record/baseline_cifar_wrn28_second.npz')['train_recall']
train3 = np.load('bin/record/baseline_cifar_wrn28_third.npz')['train_recall']
plt.plot(np.arange(EPOCH), np.concatenate((train1,train2,train3)), label='train recall')
plt.legend()

'''
for i in range(1, 5):
    recall1 = np.load('bin/record/baseline_cifar_error_0.' + str(i) + '_gaussian_0.1_first.npz')['val_recall']
    recall2 = np.load('bin/record/baseline_cifar_error_0.' + str(i) + '_gaussian_0.1_second.npz')['val_recall']
    recall3 = np.load('bin/record/baseline_cifar_error_0.' + str(i) + '_gaussian_0.1_third.npz')['val_recall']
    plt.plot(np.arange(EPOCH), np.concatenate((recall1,recall2,recall3)), label=str(i + 1) + '0% noise')
    plt.legend()
'''

plt.xlabel('Epoch')
plt.ylabel('Recall Accuracy')
plt.show()
