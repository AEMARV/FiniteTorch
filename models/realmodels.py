from definition import *
from optstructs import *
import torch
import torch.nn
''' Available Options for KLCONV('klconv'):
	r: receptive field,
	f: number of filters,
	pad: padding amount either 'same' or 'valid'
	bias: does it have bias (Bias not implemented yet)
	
	param: selects the parameterization methods including->
	
	 1. 'logstochu': log parameterization, stochastic gradient
	 and all dists are initialized to uniform
	 2. 'logstoch' : log parameterizaiton, stochastic gradient
	 3. 'log' : log parametrizaiton, exponential dists initialization  
	
	
	
	'''

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
	                           gpu=True)
	''' Create All opts'''
	opts = allOpts(netopts=opts_net,
	               optimizeropts=opts_optim,
	               epocheropts=opts_epocher,
	               )
	return opts
def quick_cifar_hello_kl_stoch() -> allOpts:
	model_string = ''
	nl = 'lnorm|s:1'
	d = '->'
	finish = 'fin'
	model_string += 'klconvb|r:5,f:32,pad:same,bias:1,param:logstochu' + nl +d
	model_string += 'klavgpool|r:3,f:32,pad:same,stride:2,bias:1' + d
	model_string += 'klconv|r:5,f:64,pad:same,bias:1,param:logstochu' + d + nl + d
	model_string += 'klavgpool|r:3,f:32,pad:same,stride:2,bias:1' + d
	model_string += 'klconv|r:4,f:64,pad:same,bias:1,param:logstochu' + d + nl + d
	model_string += 'klconv|r:7,f:10,pad:valid,bias:1,param:logstochu' +d +nl + d
	model_string += finish

	''' Net Options'''
	opts_net = NetOpts(model_string,
	                   inputchannels=3,
	                   inputspatsz=32)
	'''Optimizer Options'''
	opts_optim =OptimOpts(lr=.01,
	                      type='SGD',
	                      momentum=0.9,
	                      weight_decay=0,
	                      dampening=0,
	                      nestrov=False)
	'''Epocher Options'''
	opts_epocher = EpocherOpts(epochnum=150,
	                           batchsz=100,
	                           shuffledata=True,
	                           loss=torch.nn.NLLLoss(),
	                           numworkers=1,
	                           gpu=True)
	''' Create All opts'''
	opts = allOpts(netopts=opts_net,
	               optimizeropts=opts_optim,
	               epocheropts=opts_epocher,
	               )
	return opts
def quick_cifar_hello_kl() -> allOpts:
	model_string = ''
	d = '->'
	nl = 'lnorm|s:0'
	param = 'dirichstochunif'
	finish = 'fin'
	model_string += 'klconvb|r:3,f:64,pad:same,bias:1,param:{},stoch:0'.format(param) + d+'lnorm|s:1'+ d
	model_string += 'klconv|r:3,f:64,pad:same,bias:1,param:{},stoch:0'.format(param) + d + 'lnorm|s:1' + d
	model_string += 'klconv|r:3,f:64,pad:same,bias:1,param:{},stoch:0'.format(param) + d + 'lnorm|s:1' + d
	model_string += 'klavgpool|r:3,f:32,pad:same,stride:2,bias:1' + d
	model_string += 'klconv|r:3,f:128,pad:same,bias:1,param:{},stoch:0'.format(param) + d + 'lnorm|s:1' + d
	model_string += 'klconv|r:3,f:128,pad:same,bias:1,param:{},stoch:0'.format(param) + d + 'lnorm|s:1' + d
	model_string += 'klconv|r:3,f:128,pad:same,bias:1,param:{},stoch:0'.format(param) + d + 'lnorm|s:1' + d
	model_string += 'klavgpool|r:3,f:32,pad:same,stride:2,bias:1' + d
	model_string += 'klconv|r:3,f:256,pad:same,bias:1,param:{},stoch:0'.format(param) + d + 'lnorm|s:1' + d
	model_string += 'klconv|r:3,f:256,pad:same,bias:1,param:{},stoch:0'.format(param) + d + 'lnorm|s:1' + d
	model_string += 'klconv|r:3,f:256,pad:same,bias:1,param:{},stoch:0'.format(param) + d + 'lnorm|s:1' + d
	model_string += 'klavgpool|r:3,pad:same,stride:2,bias:1' + d
	model_string += 'klconv|r:4,f:10,pad:valid,bias:1,param:{},stoch:0'.format(param) +d +'lnorm|s:1' + d
	model_string += finish

	''' Net Options'''
	opts_net = NetOpts(model_string,
	                   inputchannels=3,
	                   inputspatsz=32)
	'''Optimizer Options'''
	opts_optim =OptimOpts(lr=1,
	                      type='SGD',
	                      momentum=0.9,
	                      weight_decay=0,
	                      dampening=0,
	                      nestrov=False)
	'''Epocher Options'''
	opts_epocher = EpocherOpts(epochnum=3000,
	                           batchsz=20,
	                           shuffledata=True,
	                           loss=torch.nn.NLLLoss(),
	                           numworkers=1,
	                           gpu=True)
	''' Create All opts'''
	opts = allOpts(netopts=opts_net,
	               optimizeropts=opts_optim,
	               epocheropts=opts_epocher,
	               )
	return opts
def quick_cifar_hello_kl_v1() -> allOpts:
	model_string = ''
	d = '->'
	nl = 'lnorm|s:0'
	convparams = 'param:logstoch,stoch:0'
	finish = 'fin'
	model_string += 'klconvb|r:3,f:64,pad:same,bias:1,{}'.format(convparams) + d + nl + d
	model_string += 'klconv|r:3,f:64,pad:same,bias:1,{}'.format(convparams)  + d + nl + d
	model_string += 'klconv|r:3,f:64,pad:same,bias:1,{}'.format(convparams)  + d + nl + d
	model_string += 'klavgpool|r:3,f:32,pad:same,stride:2,bias:1,stoch:1'    + d + nl + d
	model_string += 'klconv|r:3,f:128,pad:same,bias:1,{}'.format(convparams) + d + nl + d
	model_string += 'klconv|r:3,f:128,pad:same,bias:1,{}'.format(convparams) + d + nl + d
	model_string += 'klconv|r:3,f:128,pad:same,bias:1,{}'.format(convparams) + d + nl + d
	model_string += 'klavgpool|r:3,f:32,pad:same,stride:2,bias:1,stoch:1'    + d + nl + d
	model_string += 'klconv|r:3,f:256,pad:same,bias:1,{}'.format(convparams) + d + nl + d
	model_string += 'klconv|r:3,f:256,pad:same,bias:1,{}'.format(convparams) + d + nl + d
	model_string += 'klconv|r:3,f:256,pad:same,bias:1,{}'.format(convparams) + d + nl + d
	model_string += 'klavgpool|r:3,pad:same,stride:2,bias:1,stoch:1'         + d + nl + d
	model_string += 'klconv|r:1,f:10,pad:valid,bias:1,{}'.format(convparams) + d + nl + d
	model_string += 'klavgpool|r:4,pad:valid,stride:1,bias:1,stoch:1' + d + nl + d
	model_string += finish

	''' Net Options'''
	opts_net = NetOpts(model_string,
	                   inputchannels=3,
	                   inputspatsz=32)
	'''Optimizer Options'''
	opts_optim =OptimOpts(lr=1,
	                      type='SGD',
	                      momentum=0.9,
	                      weight_decay=0,
	                      dampening=0,
	                      nestrov=False)
	'''Epocher Options'''
	opts_epocher = EpocherOpts(epochnum=3000,
	                           batchsz=100,
	                           shuffledata=True,
	                           loss=torch.nn.NLLLoss(),
	                           numworkers=1,
	                           gpu=True)
	''' Create All opts'''
	opts = allOpts(netopts=opts_net,
	               optimizeropts=opts_optim,
	               epocheropts=opts_epocher,
	               )
	return opts
def quick_cifar_hello_kl_v2() -> allOpts:
	model_string = ''
	d = '->'
	nl = 'lnorm|s:1'
	convparams = 'param:sphere,stoch:0'
	finish = 'fin'
	model_string += 'klconvb|r:3,f:64,pad:same,bias:1,{}'.format(convparams) + d + nl + d
	model_string += 'klconv|r:3,f:64,pad:same,bias:1,{}'.format(convparams)  + d + nl + d
	model_string += 'klconv|r:3,f:64,pad:same,bias:1,{}'.format(convparams)  + d + nl + d
	model_string += 'klavgpool|r:3,f:32,pad:same,stride:2,bias:1,stoch:1'    + d + nl + d
	model_string += 'klconv|r:3,f:128,pad:same,bias:1,{}'.format(convparams) + d + nl + d
	model_string += 'klconv|r:3,f:128,pad:same,bias:1,{}'.format(convparams) + d + nl + d
	model_string += 'klconv|r:3,f:128,pad:same,bias:1,{}'.format(convparams) + d + nl + d
	model_string += 'klavgpool|r:3,f:32,pad:same,stride:2,bias:1,stoch:1'    + d + nl + d
	model_string += 'klconv|r:3,f:256,pad:same,bias:1,{}'.format(convparams) + d + nl + d
	model_string += 'klconv|r:3,f:256,pad:same,bias:1,{}'.format(convparams) + d + nl + d
	model_string += 'klconv|r:3,f:256,pad:same,bias:1,{}'.format(convparams) + d + nl + d
	model_string += 'klavgpool|r:3,pad:same,stride:2,bias:1,stoch:1'         + d + nl + d
	model_string += 'klconv|r:1,f:10,pad:valid,bias:1,{}'.format(convparams) + d + nl + d
	model_string += 'klavgpool|r:4,pad:valid,stride:1,bias:1,stoch:1' + d + nl + d
	model_string += finish

	''' Net Options'''
	opts_net = NetOpts(model_string,
	                   inputchannels=3,
	                   inputspatsz=32)
	'''Optimizer Options'''
	opts_optim =OptimOpts(lr=1,
	                      type='SGD',
	                      momentum=0.9,
	                      weight_decay=0,
	                      dampening=0,
	                      nestrov=False)
	'''Epocher Options'''
	opts_epocher = EpocherOpts(epochnum=3000,
	                           batchsz=100,
	                           shuffledata=True,
	                           loss=torch.nn.NLLLoss(),
	                           numworkers=1,
	                           gpu=True)
	''' Create All opts'''
	opts = allOpts(netopts=opts_net,
	               optimizeropts=opts_optim,
	               epocheropts=opts_epocher,
	               )
	return opts
def histog_kl_v0() -> allOpts:
	model_string = ''
	d = '->'
	nl = 'lnorm|s:1'
	convparams = 'param:log,stoch:0'
	finish = 'fin'
	model_string += 'klconvb|r:3,f:64,pad:same,bias:1,{}'.format(convparams) + d + nl + d
	model_string += 'klavgpool|r:3,f:32,pad:same,stride:1,bias:1,stoch:1' + d + nl + d
	model_string += 'klconv|r:1,f:64,pad:same,bias:1,{}'.format(convparams)  + d + nl + d
	model_string += 'klavgpool|r:3,f:32,pad:same,stride:1,bias:1,stoch:1' + d + nl + d
	model_string += 'klconv|r:1,f:64,pad:same,bias:1,{}'.format(convparams)  + d + nl + d
	model_string += 'klavgpool|r:3,f:32,pad:same,stride:2,bias:1,stoch:1'    + d + nl + d
	model_string += 'klconv|r:1,f:128,pad:same,bias:1,{}'.format(convparams) + d + nl + d
	model_string += 'klavgpool|r:3,f:32,pad:same,stride:1,bias:1,stoch:1' + d + nl + d
	model_string += 'klconv|r:1,f:128,pad:same,bias:1,{}'.format(convparams) + d + nl + d
	model_string += 'klavgpool|r:3,f:32,pad:same,stride:1,bias:1,stoch:1' + d + nl + d
	model_string += 'klconv|r:1,f:128,pad:same,bias:1,{}'.format(convparams) + d + nl + d
	model_string += 'klavgpool|r:3,f:32,pad:same,stride:2,bias:1,stoch:1'    + d + nl + d
	model_string += 'klconv|r:1,f:256,pad:same,bias:1,{}'.format(convparams) + d + nl + d
	model_string += 'klavgpool|r:3,f:32,pad:same,stride:1,bias:1,stoch:1' + d + nl + d
	model_string += 'klconv|r:1,f:256,pad:same,bias:1,{}'.format(convparams) + d + nl + d
	model_string += 'klavgpool|r:3,f:32,pad:same,stride:1,bias:1,stoch:1' + d + nl + d
	model_string += 'klconv|r:1,f:256,pad:same,bias:1,{}'.format(convparams) + d + nl + d
	model_string += 'klavgpool|r:3,pad:same,stride:2,bias:1,stoch:1'         + d + nl + d
	model_string += 'klconv|r:1,f:10,pad:same,bias:1,{}'.format(convparams) + d + nl + d
	model_string += 'klavgpool|r:4,pad:valid,stride:1,bias:1,stoch:1' + d + nl + d
	model_string += finish

	''' Net Options'''
	opts_net = NetOpts(model_string,
	                   inputchannels=3,
	                   inputspatsz=32)
	'''Optimizer Options'''
	opts_optim =OptimOpts(lr=1,
	                      type='SGD',
	                      momentum=0.9,
	                      weight_decay=0,
	                      dampening=0,
	                      nestrov=False)
	'''Epocher Options'''
	opts_epocher = EpocherOpts(epochnum=3000,
	                           batchsz=100,
	                           shuffledata=True,
	                           loss=torch.nn.NLLLoss(),
	                           numworkers=1,
	                           gpu=True)
	''' Create All opts'''
	opts = allOpts(netopts=opts_net,
	               optimizeropts=opts_optim,
	               epocheropts=opts_epocher,
	               )
	return opts