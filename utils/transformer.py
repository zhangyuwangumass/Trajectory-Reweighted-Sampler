import torch
import torch.nn.functional as F
import numpy as np
import numpy.random as random

class Transformer:

    def get_random_crop_params(self, img, output_size):
        c, w, h = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, i + th, j + tw

    def random_crop(self, img, size, padding=0):
        if padding > 0:
            C, H, W = img.shape
            new_img = np.zeros((C, H+2*padding, W+2*padding))
            new_img[:,padding:H+padding,padding:W+padding] = img
        i, j, h, w = self.get_random_crop_params(new_img, size)

        return new_img[:, i:h, j:w]

    def random_horizontal_flip(self, img, p=0.5):
        if random.random() < p:
            return np.flip(img, axis=2)
        return img



