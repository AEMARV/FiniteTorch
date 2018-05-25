from optstructs import *
class ResultStruct(object):
	def __init__(self, optsall: allOpts):
		self.optsAll = optsall
		self.val_acc = []
		self.val_loss = []
		self.train_acc = []
		self.train_loss = []

	def add_epoch_res(self, train_acc, train_loss, val_acc, val_loss):
		self.val_acc += [val_acc]
		self.val_loss += [val_loss]
		self.train_acc += [train_acc]
		self.train_loss += [train_loss]

	def draw(self):
