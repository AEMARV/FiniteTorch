import torch
import klconvs
import torch.nn.functional as F
from layers.klfunctions import LogSumExp
import time
from torch import Tensor
dev = torch.device("cuda:0")
testinput = torch.rand(100,128,32,32).to(dev)
testfilt = torch.rand(100,128,3,3).to(dev)**100
testfilt = testfilt / testfilt.sum(dim=1,keepdim=True)
testfilt = (testfilt.log())
LSE = LogSumExp
#testfilt = testfilt - LSE.apply(testfilt,1)
temp = 0
myout= 0
elapsed=0
elapsedtorch=0
temptime =0
pad = torch.ones(4,1).to(dev)
trials = 100
for i in range(trials):
	t = time.time()
	temp = klconvs.forward(testinput, testfilt, pad)
	temptime = time.time() - t
	myout = myout + temp
	elapsed = elapsed + temptime

	t = time.time()
	torchout = F.conv2d(testinput, testfilt.exp(), bias=None, stride=1, padding=1)
	temptime = time.time() - t
	elapsedtorch = elapsedtorch + temptime



myout = myout / trials
elapsed = elapsed/trials
elapsedtorch = elapsedtorch/trials
print('Elpased time Myconv: ',elapsed, 'Elpased time torchconv: ',elapsedtorch)
print('L2 diff: ',((myout-torchout)**2).mean().item())
#print('L2 diff: ',((torchout-torchout)**2).mean().item())
#print('L2 diff: ',((myout-myout)**2).mean().item())
print('is distribution?: ',((testfilt.exp()).sum(dim=1,keepdim=False)).mean().item())
#print('output size: ',(torchout).shape)