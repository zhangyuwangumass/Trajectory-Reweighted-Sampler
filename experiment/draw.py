import numpy as np
import matplotlib.pyplot as plt

COMP = 6

his = np.load('bin/record/pretrain_shallow.npz')['his']
train_recall = np.load('bin/record/pretrain_shallow.npz')['train_recall']
val_recall = np.load('bin/record/pretrain_shallow.npz')['val_recall']

NUM, EPOCH = his.shape
class_index = np.load('bin/record/pretrain_shallow.npz')['index']

mu = np.load('bin/record/stats_shallow.npz')['mu']
var = np.load('bin/record/stats_shallow.npz')['var']
pred = np.load('bin/record/stats_shallow.npz')['pred']



fig = plt.figure()

for i in range(COMP):
    mask = np.where(pred==i)
    index = class_index[mask]
    total = index.shape[0]
    error = np.array(np.where(index==1)).shape[1]
    gaussian = np.array(np.where(index==-1)).shape[1]
    error_rate = round(error / max(1,total), 3)
    gaussian_rate = round(gaussian / max(1,total), 3)
    

    ax1 = fig.add_subplot(2,3,i+1)
    ax1.set_title('cluster ' + str(i))
    ax1.set_ylim(0,0.3)
    ax2 = ax1.twinx()
    ax2.set_ylim(0,NUM//COMP)
    ax2.set_yticks([])
    ax2.bar([EPOCH // 2 + 1], [total], fc='y', label='total = ' + str(total))
    ax2.bar([EPOCH // 2 - 1], [error], fc='r', label='error = ' + str(error) + ', ' + str(error_rate))
    ax2.bar([EPOCH // 2], [gaussian], fc='b', label='gaus = ' + str(gaussian) + ', ' + str(gaussian_rate))
    ax2.legend(loc=4)

    ax1.errorbar(np.arange(EPOCH), mu[i,:], yerr=var[i,:])
    ax1.plot(np.arange(EPOCH), mu[i,:])

plt.show()

'''
data = np.load('bin/data/data.npz')['data']

print(data.shape)
print(pred.shape)

count = 1
for i in range(COMP):
    mask = np.where(pred == i)
    imgs = data[mask]
    print(imgs.shape)
    imgs = imgs[500:510]
    for img in imgs:
        label = img[-2]
        img = img[:-2]
        plt.subplot(6, 10, count)
        plt.imshow(img.reshape((28, 28)))
        plt.title(str(label))
        # plt.get_xaxis().set_visible(False)
        # plt.get_yaxis().set_visible(False)
        count += 1

        # fig.tight_layout(pad=0.1)

plt.show()
'''
'''


plt.plot(np.arange(EPOCH), train_recall, label="train recall")
plt.plot(np.arange(EPOCH), val_recall, label="val recall")
plt.legend()

plt.show()
'''
