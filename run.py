import torch
import torchvision as tv
import torch.nn as nn
from models.modelutils import *
from data.datasetutils import *
if __name__ == '__main__':
	model_names = ['']
	dataset_names = ['']
	for dataset_name in dataset_names:
		for model_name in model_names:
			#TODO: Create Iterable for Training
			trainset,testset, opts = create_data_set(dataset_name)
			#TODO: Get Model Module and Optimizer
			model_module, opts = get_model_module(model_name, opts)
			#TODO: Train and Validate



