# Trajectory-Reweighted-Sampler
This is the research project for CS689 course.<br><br>

In this project we are trying to classify data points into different classes by clustering their training trajectories. This comes from both down-sampling noisy data points (https://arxiv.org/abs/1803.09050) and selecting data points with different difficulties in different stages of training (https://arxiv.org/abs/1704.07433). We propose that noisy, easy and difficult data points are just data points with different training significance to the model, and the significance can be measured by comparing their gradient contribution with a small given validation set. Additionally, in order to deal with overfitting on the validation set, we propose to use trajectory clustering to reduce the significance estimation from point-wise to cluster-wise, increasing the model's generalization. Finally, we reweight the number of samples from each cluster in the mini-batch 
proportional to their significance.<br><br>

We experimented on some standard baseline tasks like image classification on MNIST and CIFAR, and reported consistent improvement on noisy datasets.
