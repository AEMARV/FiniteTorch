import torch as t
import torchvision as tv
from definition import *
def create_data_set(data_set_name):
	opts = create_opts_struct()
	if data_set_name is 'cifar10':
		train_loader,test_loader = get_cifar10()
		opts[OPTS_DATASET][OPTS_DATASET_CHANS] = 3
		opts[OPTS_DATASET][OPTS_DATASET_SPATIAL] = 32
	return train_loader,test_loader,opts

def get_cifar10():
	trainset = tv.datasets.CIFAR10(PATH_DATA, train=True, download=True)
	testset = tv.datasets.CIFAR10(PATH_DATA, train=False, download=True)
	train_loader = t.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, sampler=None, num_workers=1)
	test_loader = t.utils.data.DataLoader(testset, batch_size=100, shuffle=True, sampler=None, num_workers=1)
	return train_loader,test_loader
def create_opts_struct():
	opts = {}
	opts[OPTS_DATASET] = {}
	opts[OPTS_OPTMIZER] = {}
	return opts