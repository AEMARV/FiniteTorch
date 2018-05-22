class Epocher(object):
	def __init__(self,model=None,
	             trainloader=None,
	             testloader=None,
	             optimizer=None,
	             opts=None,
	             epochnum=):

		super(Epocher, self).__init__()
		self.model = model
		self.trainloader= trainloader
		self.testloader = testloader
		self.optimizer = optimizer
		self.opts = opts

	def run_epoch(self):
		run_loss = 0.0
		# TODO: Train on batches
		for i in range(self.epochnum):
		# TODO: Evaluate Test
		# TODO: Visualizing
		# TODO: Saving Results

	def run_many_epochs(self,epochnum):
		for epoch in range(epochnum):
			self.run_epoch()