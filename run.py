from models.modelutils import *
from data.datasetutils import *
from trainvalid.optimizer import *
from trainvalid.epocher import *
if __name__ == '__main__':
	model_names = ['hello_stoch_hello_kl']
	dataset_names = ['cifar10']
	for dataset_name in dataset_names:
		for model_name in model_names:
			# Get Model Module and Optimizer
			model_module, opts = get_model_module(model_name)
			# Create Iterable for Training
			trainset, testset, opts = create_data_set(dataset_name,opts)
			# Create Optimizer
			optmizer = create_optimizer(opts,model_module)
			#TODO: PRINT OPTIONS
			#TODO: Train and Validate model
			epocher = Epocher(model_module,
                              optmizer,
			                  trainset,
			                  testset,
			                  opts.epocheropts,
			                  opts)
			epocher.run_many_epochs()


