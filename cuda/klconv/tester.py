import torch
import klconvs
import torch.nn.functional as F
from layers.klfunctions import LogSumExp
import time
from layers.klfunctions import KLConvStoch
import typing
from torch import Tensor


dev = torch.device("cuda:0")
testinput = torch.rand(100,2,32,32,dtype=torch.float32).to(dev) # type: Tensor
testfilt = torch.rand(100,2,3,3,dtype=torch.float32).to(dev) # type: Tensor
testfilt = testfilt / testfilt.sum(dim=1,keepdim=True)
testfilt = (testfilt.log())
testfilt.detach()
testinput.detach()
test_filt_mine = testfilt.clone()
test_input_mine = testinput.clone()
test_filt_mine.detach()
test_input_mine.detach()
testfilt.requires_grad_(True)
testinput.requires_grad_(True)
test_filt_mine.requires_grad_(True)
test_input_mine.requires_grad_(True)
LSE = LogSumExp
#testfilt = testfilt - LSE.apply(testfilt,1)
temp = 0
myout= 0
elapsed=0
elapsedtorch=0
elapsed_back_mine = 0
temptime =0
pad = torch.ones(4,1).to(dev)
trials = 100
dzdout,rand_sample =klconvs.forward(testinput, testfilt)
dzdout = dzdout *0 + 1
dzdin_mine = 0
dzdfilt_mine =0
for i in range(trials):
	t = time.time()

	''' Mine '''
	#temp,rand_sample = klconvs.forward(test_input_mine, test_filt_mine)
	myout= KLConvStoch.apply(test_input_mine, test_filt_mine)
	loss_mine = (myout.sum())
	tin, tfilt=klconvs.backward_rand(dzdout,test_input_mine,test_filt_mine)
	dzdin_mine += tin
	dzdfilt_mine += tfilt
	#loss_mine.backward()
	temptime = time.time() - t
	myout = myout + temp
	elapsed += temptime


	''' His'''
	t= time.time()
	torchout = KLConvStoch.normal_klconv(testinput,testfilt)
	#torchout = F.conv2d(testinput, testfilt.exp(), bias=None, stride=1, padding=1)
	loss = torchout.sum()  # type: Tensor
	loss.backward()


	temptime = time.time() - t
	elapsedtorch = elapsedtorch + temptime


#myout = myout/ trials
dzdin_mine =  dzdin_mine /trials
dzdfilt_mine = dzdfilt_mine/ trials
dzdin_torch = testinput.grad / trials
dzdfilt_torch = testfilt.grad /trials
elapsed = elapsed/trials
elapsedtorch = elapsedtorch/trials
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


