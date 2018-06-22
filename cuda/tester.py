import torch
import klconvs
import torch.nn.functional as F
from layers.klfunctions import LogSumExp
from torch import Tensor
dev = torch.device("cuda:0")
testinput = torch.rand(100,3,32,32).to(dev)
testfilt = torch.rand(128,3,3,3).to(dev)
testfilt = testfilt / testfilt.sum(dim=1,keepdim=True)
testfilt = (testfilt.log())
LSE = LogSumExp
#testfilt = testfilt - LSE.apply(testfilt,1)
temp = 0
myout= 0
pad = torch.ones(4,1).to(dev)
for i in range(1):
	temp = klconvs.forward(testinput,testfilt,pad)
	myout = ((i)*myout + temp)/(i+1)
torchout = F.conv2d(testinput,testfilt.exp(),bias=None,stride=1,padding=0)

print(((myout[0:,0:,0:-2,0:-2]-torchout)**2).mean())
print(((testfilt.exp()).sum(dim=1,keepdim=False)).mean())
print((torchout).shape)