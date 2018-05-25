from torch.nn import Module
import torch.nn as nn
from torch.utils.data import DataLoader
from optstructs import EpocherOpts
from torch.optim import Optimizer
import torch
class Epocher(object):
	def __init__(self,model:Module,
				 optimizer:Optimizer,
				 trainloader:DataLoader,
				 testloader:DataLoader,
				 opts:EpocherOpts):

		super(Epocher, self).__init__()
		self.model = model
		self.optimizer = optimizer
		self.trainloader= trainloader
		self.testloader = testloader
		self.optimizer = optimizer
		self.opts = opts

	def run_epoch(self,prefixprint:str):
		run_loss = 0.0
		totalbatches = 100
		corrects = 0
		totalsamples = 0
		# TODO: Train on batches
		for batch_n,data in enumerate(self.trainloader):
			inputs, labels = data
			inputs, labels = inputs.to(self.opts.device),labels.to(self.opts.device)
			self.optimizer.zero_grad()
			output = self.model(inputs)
			output = output.view(-1,self.opts.classnum)
			loss = self.opts.loss(output,labels)
			predlab = torch.argmax(output,1,keepdim=False)
			accthis = (predlab==labels).sum().item()
			corrects += accthis
			totalsamples += labels.size(0)
			loss.backward()
			self.optimizer.step()
			run_loss += loss.item()
			#TODO: Print Batch Statistics
			acc = corrects/totalsamples
			print(prefixprint +':' 
				  'batch:%d/%d'%(batch_n,totalbatches) +
				  ' loss:%.4f'% (run_loss/(batch_n+1)) +
			      'train accuracy: %.2f'% (acc*100)
			      )

		# TODO: Evaluate Test
		run_loss = 0
		totalsamples = 0
		corrects = 0
		for batch_n,data in enumerate(self.trainloader):
			inputs, labels = data
			inputs, labels = inputs.to(self.opts.device),labels.to(self.opts.device)
			self.optimizer.zero_grad()
			output = self.model(inputs)
			output = output.view(-1, self.opts.classnum)
			predlab = torch.argmax(output, 1, keepdim=False)
			accthis = (predlab == labels).sum().item()
			corrects += accthis
			totalsamples += labels.size(0)
			val_acc = corrects/totalsamples
			loss = self.opts.loss(output,labels)
			run_loss += loss.item()
			avg_loss = run_loss/(batch_n+1)
			#TODO: Print Batch Statistics
		print('prefixprint:' +
			  'val_loss:%.4f' % avg_loss +
		      'val_acc:%.2f' % val_acc)

		# TODO: Visualizing
		# TODO: Saving Results

	def run_many_epochs(self):
		for epoch in range(self.opts.epochnum):
			prefixtext = 'Epoch %d' % epoch
			self.run_epoch(prefixtext)