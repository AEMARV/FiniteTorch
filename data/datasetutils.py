import torch as t
import torchvision as tv
from optstructs import *
from definition import *
import torchvision.transforms as transforms
def create_data_set(data_set_name:str,opts:allOpts):
	if data_set_name is 'cifar10':
		train_loader,test_loader = get_cifar10(opts)
	else:
		raise('Dataset Not Found: ' + data_set_name)
	return train_loader,test_loader,opts

def get_cifar10(opts:allOpts):
	# Obtain options from opts class
	batchsz = opts.epocheropts.batchsz
	isshuffle = opts.epocheropts.shuffledata
	opts.epocheropts.classnum=10

	transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	# Construct loaders
	trainset = tv.datasets.CIFAR10(PATH_DATA, train=True, download=True,transform=transform)
	testset = tv.datasets.CIFAR10(PATH_DATA, train=False, download=True,transform=transform)
	train_loader = t.utils.data.DataLoader(trainset, batch_size=batchsz, shuffle=isshuffle, sampler=None, num_workers=1)
	test_loader = t.utils.data.DataLoader(testset, batch_size=batchsz, shuffle=isshuffle, sampler=None, num_workers=1)
	return train_loader,test_loader