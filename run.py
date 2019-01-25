from models.modelutils import *
from data.datasetutils import *
from trainvalid.optimizer import *
from trainvalid.epocher import *
from experiment.Experiments import *

def boolprompt(question):
	answer=''
	while(answer.lower!='n' or answer.lower()!='y'):
		answer = input(question+' [y]/[n]')
		if answer[0].lower()=='y':
			return True
		elif answer[0].lower()=='n':
			return False
		else:
			print('please answer with y/n characters')



if __name__ == '__main__':
	#exp = Quickie(1)
	#exp = Synthetic_PMaps(1)
	exp = QuickCifar(1)
	exp.run()
	# experiment_name = 'Baselines'
	# model_names = ['quick_cifar']
	# dataset_names = ['cifar100']
	# save_result = boolprompt('Do you want to save the results?')
	# epocheropts = EpocherOpts(save_result,
	#                           epochnum=150,
	#                           batchsz=100,
	#                           shuffledata=True,
	#                           numworkers=1,
	#                           gpu=True)
	# for dataset_name in dataset_names:
	# 	for model_name in model_names:
	# 		xp_dscrpt = [experiment_name,dataset_name,model_name]
	# 		opts = create_opts(experiment_name,dataset_name,model_name,epocheropts)
	# 		epocher = Epocher(opts)
	# 		epocher.run_many_epochs()

