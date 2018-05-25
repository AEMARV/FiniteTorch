import torch
class EpocherOpts(object):
	def __init__(self,
	             epochnum=150,
	             batchsz=100,
	             shuffledata=True,
	             batchperepoch=500,
	             numworkers=1,
	             loss=None,
	             gpu=True):
		self.epochnum = epochnum
		self.batchsz = batchsz
		self.shuffledata = shuffledata
		self.numworkers = numworkers
		self.batchperepoch = batchperepoch
		self.loss = loss
		self.gpu = gpu
		self.classnum = 0
		if self.gpu:
			self.device = torch.device("cuda:0")
		else:
			self.device = torch.device("cpu")


class NetOpts(object):
	def __init__(self,modelstring,
	             inputchannels=3,
	             inputspatsz=32,
	             convinit=None,
				 convbinit=None,
				 biasinit=None,
				 convreg=None,
				 biasreg=None,
				 ):
		self.modelstring = modelstring
		self.inputspatsz=inputspatsz
		self.inputchannels=inputchannels
		self.convinit =convinit
		self.convbinit = convbinit
		self.biasinit = biasinit
		self.biasreg = biasreg
		self.convreg = convreg
class OptimOpts(object):
	def __init__(self,lr=1,
	             type='SGD',
	             momentum=0.9,
	             weight_decay=0,
	             dampening=0,
	             nestrov=False,
				 ):
		self.lr = lr
		self.type = type
		self.momentum = momentum
		self.weight_decay = weight_decay
		self.dampening = dampening
		self.nestrov = nestrov



class allOpts(object):
	def __init__(self,netopts=NetOpts(''),optimizeropts=None,epocheropts=EpocherOpts(),gpu=True):
		self.netopts = netopts
		self.optimizeropts = optimizeropts
		self.epocheropts = epocheropts
		self.gpu=gpu

