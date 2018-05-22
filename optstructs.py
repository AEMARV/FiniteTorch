class allOpts(object):
	def __init__(self,netopts=None,optimizeropts=None,epocheropts=None):
		self.netopts = netopts
		self.optimizeropts = optimizeropts
		self.epocheropts = epocheropts


class EpocherOpt(object):
	def __init__(self,epochnum=-1,batchsz=-1):
		self.epochnum = epochnum
		self.batchsz = batchsz
class NetOpts(object):
	def __init__(self,):