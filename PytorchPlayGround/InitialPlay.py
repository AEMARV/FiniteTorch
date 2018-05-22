from __future__ import print_function
import torchvision as tv
import torch.nn as nn
import torch.utils.data
import torch as t
from definition import *
if __name__ == '__main__':
    trainset = tv.datasets.CIFAR10(PATH_DATA,train=True,download=True)
    testset = tv.datasets.CIFAR10(PATH_DATA, train=False, download=True)
    train_loader = t.utils.data.DataLoader(trainset,batch_size=100,shuffle=True,sampler=None,num_workers=1)
    test_loader = t.utils.data.DataLoader(testset,batch_size=100,shuffle=True,sampler=None,num_workers=1)
