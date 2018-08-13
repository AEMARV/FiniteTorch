from definition import *
from optstructs import *
import torch
import torch.nn
from torch.optim.lr_scheduler import *
from layers.klfunctions import LossExpectedError
from trainvalid.lr_schedulers import *
import torchvision.transforms as transforms

''' Relu vs Sigmoid Experiments:
	for VGG-QuickCifar-NIN the effects of both nonlinearities are tested.


	'''


def quick_cifar(classnum) -> allOpts:
	model_string = ''
	nl = 'relu'
	d = '->'
	finish = 'fin'
	model_string += 'klconv|r:5,f:32,pad:same,bias:1,isrelu:1,stoch:0' + d
	model_string += 'klavgpool|r:3,f:32,pad:same,stride:2,bias:1' + d
	model_string += 'klconv|r:5,f:64,pad:same,bias:1' + d + nl + d
	model_string += 'klavgpool|r:3,f:32,pad:same,stride:2,bias:1' + d
	model_string += 'klconv|r:4,f:64,pad:same,bias:1' + d + nl + d
	model_string += 'klconv|r:7,f:'+ str(classnum) + ',pad:valid,bias:1' + d+ 'lnorm|s:0'+d
	model_string += finish

	'''LR SCHED'''
	initlr = .01
	lr_sched = exp_decay_lr(init_lr=initlr,step=20,exp_decay_perstep=1)

	''' Net Options'''
	opts_net = NetOpts(model_string,
	                   inputchannels=3,
	                   inputspatsz=32)
	'''Optimizer Options'''
	opts_optim =OptimOpts(lr=initlr,
	                      lr_sched= lr_sched,
	                      type='SGD',
	                      momentum=0.9,
	                      weight_decay=1e-5,
	                      dampening=0,
	                      nestrov=False)
	'''Epocher Options'''
	opts_epocher = EpocherOpts(epochnum=150,
	                           batchsz=100,
	                           shuffledata=True,
	                           #loss=torch.nn.CrossEntropyLoss(),
	                           loss=LossExpectedError.apply,
	                           numworkers=1,
	                           gpu=True)
	''' Create All opts'''
	opts = allOpts(netopts=opts_net,
	               optimizeropts=opts_optim,
	               epocheropts=opts_epocher,
	               )
	return opts