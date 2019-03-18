import numpy as np
import random

class Sampler:
    component = 0
    index = []
    sample_weight = None

    def __init__(self, index, component):
        self.component = component
        for i in range(component):
            self.index.append(np.where(index==i)[0])
        self.sample_weight = np.ones((component,)) / component

    def set_weight(self, sample_weight):
        self.sample_weight = sample_weight

    def adjust_weight(self, delta):
        self.sample_weight = self.sample_weight + delta
        self.sample_weight[self.sample_weight < 0] = 0
        self.sample_weight /= self.sample_weight.sum()

    def sample_train_batch(self, BATCH=64):
        sample_num = self.sample_weight * size
        print(sample_num)
        sample_list = []
        for i in range(self.component):
            if len(self.index[i]) < sample_num[i]:
                sample_list.append(list(self.index[i]))
            else:
                sample_list.append(random.sample(list(self.index[i]), int(sample_num[i])))

        return sample_list

    def sample_train_data(self, size):
        sample_num = self.sample_weight * size
        sample_list = []
        for i in range(self.component):
            if len(self.index[i]) < sample_num[i]:
                sample_list += list(self.index[i])
            else:
                sample_list += random.sample(list(self.index[i]), int(sample_num[i]))

        return sample_list

    def sample_eval_batch(self, BATCH=64):
        '''
        Sampling from clusters for pre-training evaluation
        :param index: list of lists containing data indexes for each class
        :param BATCH: intended batch size; if greater than the class volumn, return all of them
        :return: a list of lists containing selected evaluation data indexes
        '''
        sample_list = []
        for i in self.index:
            if len(i) < BATCH:
                sample_list.append(list(i))
            else:
                sample_list.append(random.sample(list(i),BATCH))

        return sample_list