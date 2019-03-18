import torch.nn as nn



class Flatten(nn.Module):

    def forward(self, x):
        N = x.shape[0]
        #print(x.view(N,-1 ).shape)
        return x.view(N,-1)

