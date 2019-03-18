import numpy as np
import torch.nn.functional as F


class Tracker:

    def track(self, scores, y):
        '''
        :param self:
        :param scores: prediction scores generated by the last layer of the model of shape (N, C)
        :param y: ground-truth labels for each data point of shape(N, )
        :return: the softmax probability of correct prediction
        '''
        y = y.cpu().numpy()
        N = y.shape[0]
        prob = F.softmax(scores, dim=1).cpu().numpy()
        return prob[np.arange(N), y]
