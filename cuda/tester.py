import torch
import klconvs
import torch.nn.functional as F
from torch import Tensor
dev = torch.device("cuda:0")
testinput = torch.ones(10,3,32,32).to(dev)
testfilt = torch.ones(20,3,5,5).to(dev)
pad = torch.ones(4,1).to(dev)
myout = klconvs.forward(testinput,testfilt,pad)
torchout = F.conv2d(testinput,testfilt,bias=None,stride=1)

#print(((myout-torchout)**2).mean())
