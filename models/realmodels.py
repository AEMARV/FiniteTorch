from definition import *
from optstructs import *
import torch
import torch.nn
def quick_cifar() -> allOpts:
	model_string = ''
	nl = 'relu'
	d = '->'
	finish = 'fin'
	model_string += 'conv|r:5,f:32,pad:same,bias:1' + d
	model_string += 'maxpool|r:3,f:32,pad:same,stride:2,bias:1' + d
	model_string += 'conv|r:5,f:64,pad:same,bias:1' + d + nl + d
	model_string += 'avgpool|r:3,f:32,pad:same,stride:2,bias:1' + d
	model_string += 'conv|r:4,f:64,pad:same,bias:1' + d + nl + d
	model_string += 'conv|r:7,f:10,pad:valid,bias:1' + d
	model_string += finish

	''' Net Options'''
	opts_net = NetOpts(model_string,
	                   inputchannels=3,
	                   inputspatsz=32)
	'''Optimizer Options'''
	opts_optim =OptimOpts(lr=0.01,
	                      type='SGD',
	                      momentum=0.9,
	                      weight_decay=1e-5,
	                      dampening=0,
	                      nestrov=False)
	'''Epocher Options'''
	opts_epocher = EpocherOpts(epochnum=150,
	                           batchsz=100,
	                           shuffledata=True,
	                           loss=torch.nn.CrossEntropyLoss(),
	                           numworkers=1,
	                           gpu=False)
	''' Create All opts'''
	opts = allOpts(netopts=opts_net,
	               optimizeropts=opts_optim,
	               epocheropts=opts_epocher,
	               )
	return opts
