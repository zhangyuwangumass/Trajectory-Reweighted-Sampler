import numpy as np
import matplotlib.pyplot as plt

class_index = np.load('bin/record/baseline_cifar_error_0.4_gaussian_0.1_first.npz')['index']

mu = np.load('bin/record/stats_baseline_cifar_error_0.4_gaussian_0.1_first.npz')['mu']
var = np.load('bin/record/stats_baseline_cifar_error_0.4_gaussian_0.1_first.npz')['var']
pred = np.load('bin/record/stats_baseline_cifar_error_0.4_gaussian_0.1_first.npz')['pred']

COMP, EPOCH = mu.shape
NUM = pred.shape[0]



fig = plt.figure()

for i in range(COMP):
    mask = np.where(pred==i)
    index = class_index[mask]
    total = index.shape[0]

    error = np.array(np.where(index==1)).shape[1]
    gaussian = np.array(np.where(index==-1)).shape[1]
    clean = total - error - gaussian

    error_rate = round(error / max(1,total), 3)
    gaussian_rate = round(gaussian / max(1,total), 3)
    clean_rate = round(clean / max(1,total), 3)
    

    ax1 = fig.add_subplot(2,3,i+1)
    ax1.set_title('cluster ' + str(i))
    ax1.set_ylim(0,0.7)
    ax2 = ax1.twinx()
    ax2.set_ylim(0,NUM//COMP)
    ax2.set_yticks([])
    ax2.bar([EPOCH // 2 - 1], [error], fc='r', label='error = ' + str(error) + ', ' + str(error_rate))
    ax2.bar([EPOCH // 2], [gaussian], fc='b', label='gauss = ' + str(gaussian) + ', ' + str(gaussian_rate))
    ax2.bar([EPOCH // 2 + 1], [clean], fc='y', label='clean = ' + str(clean) + ', ' + str(clean_rate))
    ax2.legend(loc=2)

    ax1.errorbar(np.arange(EPOCH), mu[i,:], yerr=var[i,:], label='mean and var of cluster softmax score')
    ax1.plot(np.arange(EPOCH), mu[i,:])
    ax1.legend(loc=3)

plt.show()