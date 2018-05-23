class EpocherOpts(object):
	def __init__(self,
	             epochnum=-1,
	             batchsz=-1,
	             shuffledata=True,
	             numworkers=1):
		self.epochnum = epochnum
		self.batchsz = batchsz
		self.shuffledata = shuffledata
		self.numworkers = numworkers


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

class allOpts(object):
	def __init__(self,netopts=NetOpts(),optimizeropts=None,epocheropts=EpocherOpts()):
		self.netopts = netopts
		self.optimizeropts = optimizeropts
		self.epocheropts = epocheropts


