import torch
import torch.nn.functional as F
import time
import typing
from torch import Tensor
from layers.klfunctions import jointConv
from layers.klfunctions import KLConvStoch
dev = torch.device("cuda:0")
testinput = torch.ones(1,5,3,3,dtype=torch.float32).to(dev) # type: Tensor
testfilt = torch.ones(1,5,3,3,dtype=torch.float32).to(dev) # type: Tensor
#testfilt = testfilt / testfilt.sum(dim=1,keepdim=True)
testfilt = (testfilt.log())
testinput= testinput.log()
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
temp = 0
myout= 0
elapsed_forward=0
elapsed_backward=0
trials = 100
for i in range(trials):
	temp_time = time.time()

	''' Mine '''
	myout= jointConv.apply(testinput, testfilt)
	temptime = time.time() - temp_time

	loss_mine = (myout.sum())
	elapsed_forward += temptime

	temp_time = time.time()
	loss_mine.backward()
	elapsed_backward += time.time() - temp_time

elapsed_forward = elapsed_forward/trials
elapsed_backward = elapsed_backward/trials
print('Forward: Elpased time: ',elapsed_forward, 'Elpased time backward: ',elapsed_backward)
print('is distribution?: ',((testfilt.exp()).sum(dim=1,keepdim=False)).mean().item())
#print('output size: ',(torchout).shape)


