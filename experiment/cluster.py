import numpy as np
import matplotlib.pyplot as plt

from trajectory_classifier.gmm import GaussianMixture

his = np.load('bin/record/baseline_cifar_error_0.4_first.npz')['his']
#his2 = np.load('bin/record/baseline_cifar_error_0.4_gaussian_0.1_second.npz')['his']
#his3 = np.load('bin/record/baseline_cifar_error_0.4_gaussian_0.1_third.npz')['his']

#his = np.hstack((np.hstack((his1, his2)), his3))

NUM, EPOCH = his.shape

COMP = 6

# it's possible to run incremental clustering: clip the most recent 30 epochs of trajectory,
# or does a weighted sum, or use transferred learning (cluster is learned on other bigger data set)
classifier = GaussianMixture(COMP, EPOCH)
classifier.fit(his)
mu = classifier.mu.numpy()
mu = mu.reshape(mu.shape[0], mu.shape[1])
var = classifier.var.numpy()
var = var.reshape(var.shape[0], var.shape[1])
pred = classifier.predict(his,prob=False)

np.savez('bin/record/stats_baseline_cifar_error_0.4',mu=mu,var=var,pred=pred)

