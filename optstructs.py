import torch
from torch.nn import Module
from definition import *
from torchvision.transforms import transforms
import torchvision as tv

'''Model Imports'''
class allOpts(object):
	def __init__(self,
	             name,
	             netopts=None,
	             optimizeropts=None,
	             epocheropts=None,
	             dataopts=None):
		self.name = name
		self.netopts = netopts
		self.optimizeropts = optimizeropts
		self.epocheropts = epocheropts
		self.dataopts=dataopts
	def device(self):
		return self.device
	def validateopts(self):
		if not self.netopts.inputspatszvalidator(self.dataopts.inputspatsz):
			raise Exception('Input spatial size is not compatible with the model')






class EpocherOpts(object):
	def __init__(self,
	             save_results,
	             epochnum=150,
	             batchsz=100,
	             shuffledata=True,
	             numworkers=1,
	             gpu=True):
		self.epochnum = epochnum
		self.batchsz = batchsz
		self.shuffledata = shuffledata
		self.numworkers = numworkers
		self.gpu = gpu
		self.save_results=save_results
		if self.gpu:
			self.device = torch.device("cuda:0")
		else:
			self.device = torch.device("cpu")


class NetOpts(object):
	''' Weight and bias init has the form
		lambda x: x.zero_()'''
	def __init__(self,modelstring,
	             inputspatszvalidator,
	             data_transforms=[],
	             classicNet=False,
	             weightinit=lambda x: x,
				 biasinit=lambda x: x,
				 ):
		self.inputspatszvalidator=inputspatszvalidator
		self.modelstring = modelstring
		self.classicNet=classicNet
		if classicNet and (not chck_lambda(weightinit) or not chck_lambda(biasinit) or not chck_lambda(inputspatszvalidator)) :
			raise Exception('Weight/bias init and size validators must be lambda functions\n W/B inits must be called on weight.data or param.data')
		self.weightinit=weightinit
		self.biasinit = biasinit
		self.data_transforms = data_transforms


class OptimOpts(object):
	def __init__(self,lr=1,
	             lr_sched_lambda = None,
	             type='SGD',
	             momentum=0.9,
	             weight_decay=0,
	             dampening=0,
	             nestrov=False,
	             loss=None
				 ):
		self.lr = lr
		self.lr_sched = None
		self.lr_sched_lambda=lr_sched_lambda
		self.type = type
		self.momentum = momentum
		self.weight_decay = weight_decay
		self.dampening = dampening
		self.nestrov = nestrov
		self.loss=loss

class DataOpts(object):
	def __init__(self,name
				 ):
		self.datasetname=name


		if name == 'cifar10':
			inputspatsz=32
			channelsize=3
			inputrange=(0,1)
			classnum=10

		elif name == 'cifar100':
			inputspatsz=32
			channelsize=3
			inputrange=(0,1)
			classnum=100

		else:
			raise Exception(name + ':Dataset options are not defined')
		self.inputspatsz = inputspatsz
		self.channelsize = channelsize
		self.inputrange = inputrange
		self.classnum = classnum

	def get_cifar10(self,opts: allOpts):
		# Obtain options from opts class
		batchsz = opts.epocheropts.batchsz
		isshuffle = opts.epocheropts.shuffledata
		transform = transforms.Compose(
			[transforms.ToTensor()] + opts.netopts.data_transforms)
		# Construct loaders
		trainset = tv.datasets.CIFAR10(PATH_DATA, train=True, download=True, transform=transform)
		testset = tv.datasets.CIFAR10(PATH_DATA, train=False, download=True, transform=transform)
		train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
		                                       num_workers=1)
		test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
		                                      num_workers=1)
		return train_loader, test_loader

	def get_cifar100(self, opts: allOpts):
		# Obtain options from opts class
		batchsz = opts.epocheropts.batchsz
		isshuffle = opts.epocheropts.shuffledata
		transform = transforms.Compose(
			[transforms.ToTensor()] + opts.netopts.data_transforms)
		# Construct loaders
		trainset = tv.datasets.CIFAR100(PATH_DATA, train=True, download=True, transform=transform)
		testset = tv.datasets.CIFAR100(PATH_DATA, train=False, download=True, transform=transform)
		train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
		                                       num_workers=1)
		test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
		                                      num_workers=1)
		return train_loader, test_loader
def chck_lambda(l):
	Lambda = lambda:0
	if isinstance(Lambda,type(l)):
		return True
	else:
		return False
class dummyclass(object):
	def __init__(self):
		return

