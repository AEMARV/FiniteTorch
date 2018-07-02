import torch
import klconvs
import torch.nn.functional as F
from layers.klfunctions import LogSumExp
import time
import typing
from torch import Tensor
dev = torch.device("cuda:0")
testinput = torch.rand(1,2,3,3,dtype=torch.float32).to(dev) # type: Tensor
testfilt = torch.rand(1,2,3,3,dtype=torch.float32).to(dev) **10 # type: Tensor
testfilt = testfilt / testfilt.sum(dim=1,keepdim=True)
testfilt = (testfilt.log())
testfilt.detach()
testinput.detach()
testfilt.requires_grad_(True)
testinput.requires_grad_(True)
LSE = LogSumExp
#testfilt = testfilt - LSE.apply(testfilt,1)
temp = 0
myout= 0
elapsed=0
elapsedtorch=0
elapsed_back_mine = 0
temptime =0
pad = torch.ones(4,1).to(dev)
trials = 1000
dzdout,rand_sample =klconvs.forward(testinput, testfilt)
dzdout = dzdout *0 + 1
dzdin_mine = 0
dzdfilt_mine =0
for i in range(trials):
	t = time.time()
	temp,rand_sample = klconvs.forward(testinput, testfilt)
	temptime = time.time() - t
	myout = myout + temp
	elapsed = elapsed + temptime
	t= time.time()
	dzdin_mine_temp, dzdfilt_mine_temp = klconvs.backward(dzdout, testinput, testfilt,rand_sample)
	temptime = time.time() - t
	dzdin_mine += dzdin_mine_temp
	dzdfilt_mine += dzdfilt_mine_temp
	elapsed_back_mine += temptime
	t = time.time()
	torchout = F.conv2d(testinput, testfilt.exp(), bias=None, stride=1, padding=(1,1))
	loss = torchout.sum()  # type: Tensor
	loss.backward()
	if i == 0:

		dzdin_torch = testinput.grad
		dzdfilt_torch = testfilt.grad

	temptime = time.time() - t
	elapsedtorch = elapsedtorch + temptime


myout = myout/ trials
dzdin_mine =  dzdin_mine /trials
dzdfilt_mine = dzdfilt_mine / trials
dzdin_torch/=trials
dzdfilt_torch /= trials
elapsed = elapsed/trials
elapsedtorch = elapsedtorch/trials
elapsed_back_mine /= trials
m = myout[0,0,0,0]
t = torchout[0,0,0,0]
print('Forward: Elpased time Myconv: ',elapsed, 'Elpased time torchconv: ',elapsedtorch)
print('Backward: Elpased time Myconv: ',elapsed_back_mine, 'Elpased time torchconv: ',elapsedtorch)
print('Forward: L2 diff: ',((myout[0:,0:,0:,0:]-torchout[0:,0:,0:,0:])**2).mean().item())
print('Bacward: L2 diff: Filt:',((dzdfilt_mine-dzdfilt_torch)**2).mean().item(),' Input:',((dzdin_mine-dzdin_torch)**2).mean().item())
#print('L2 diff: ',((torchout-torchout)**2).mean().item())
#print('L2 diff: ',((myout-myout)**2).mean().item())
print('is distribution?: ',((testfilt.exp()).sum(dim=1,keepdim=False)).mean().item())
#print('output size: ',(torchout).shape)