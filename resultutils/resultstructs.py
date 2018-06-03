
from optstructs import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(sum(map(ord, "aesthetics")))
sns.set()
class ResultStruct(object):
	def __init__(self, optsall: allOpts):
		sns.set()
		sns.set_style('darkgrid')
		self.optsAll = optsall
		self.resultdict = {}
		self.val_acc = []
		self.val_loss = []
		self.train_acc = []
		self.train_loss = []
		self.fig = plt.figure()
		self.axislist = []
		self.axislabels = []

	def add_epoch_res(self, train_acc, train_loss, val_acc, val_loss):
		self.val_acc += [val_acc]
		self.val_loss += [val_loss]
		self.train_acc += [train_acc]
		self.train_loss += [train_loss]
	def add_epoch_res_dict(self,resdict:dict):
		for i,key in enumerate(resdict.keys()):
			if key in self.resultdict.keys():
				self.resultdict[key].append(resdict[key])
			else:
				self.resultdict[key] = [resdict[key]]

	def clear_axis(self,axislist):
		for ax in axislist:
			ax.clear()
			ax.set_xlabel('Epoch')
	def draw(self):
		self.clear_axis(self.axislist)
		for i,key in enumerate(self.resultdict.keys()):
			res = self.resultdict[key]
			totalaxes = len(self.resultdict.keys())
			if len(self.axislist) <= i:
				caxis = self.fig.add_subplot(1,totalaxes,i+1)
				caxis.set_title(key)
				self.axislist.append(caxis)
			else:
				caxis = self.axislist[i]
				caxis.set_title(key)

			caxis.plot(range(len(res)),res)
			plt.show(block=False)
			plt.pause(0.001)




	def draw_deprecated(self):
		sns.set_style("darkgrid")
		if len(self.val_acc) != 0:
			epochs = len(self.val_acc)
			self.clear_axis(self.axislist)
			self.val_acc_ax.plot(range(epochs), self.val_acc, label='val_acc')
			self.train_acc_ax.plot(range(epochs), self.train_acc, label='train_acc')
			#plt.ylabel('Accuracy')
			#plt.xlabel('Epoch')
			#plt.axis([0,len(self.val_acc),0,100])

			#plt.draw()
			plt.show(block=False)
			plt.pause(0.001)



