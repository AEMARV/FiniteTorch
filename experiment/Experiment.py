from optstructs import *
from models.modelutils import *
from trainvalid.epocher import Epocher
from resultutils.resultstructs import ResultStruct
from definition import *
import pandas as pd
import xlsxwriter as xl
from abc import abstractmethod
def mkdirs(paths_list):
	full_path=''
	dir_exists=True
	for i in range(len(paths_list)):
		full_path = os.path.join(full_path, paths_list[i])
		if not os.path.exists(full_path):
			os.mkdir(full_path)
			dir_exists=False
	return dir_exists


class Experiment_(object):
	def __init__(self,trial,save_results=None):
		self.name=type(self).__name__

		self.opt_list = []
		self.trials = trial
		self.writer = None
		self.summary_dir = None
		self.result_dir = None
		if save_results is None:
			self.save_results= boolprompt('Do you want to save the results?')
			self.override = boolprompt('Do you want to override the results if they exist?')
		else:
			self.save_results = True
			self.override = True
		if self.save_results:
			self.summary_dir,exist = self.force_create_path(['.', 'Experiment Results', self.name])
			self.result_dir,exist = self.force_create_path(['.', 'Results', self.name])
		else:
			self.summary_dir, exist = self.force_create_path(['.', 'Experiment Results', "Temp"])
			self.result_dir, exist = self.force_create_path(['.', 'Results', "Temp"])
			self.override=True

	def get_model_opts(self, model_name: str, dataopts: DataOpts) ->Tuple[NetOpts,OptimOpts] :
		netopts, optimopts = globals()[model_name](dataopts)
		return netopts, optimopts

	def create_opts(self, expmt_name, dataset_name, model_name, epocheropts):
		dataopts = DataOpts(dataset_name)
		netopts, optimopts = self.get_model_opts(model_name, dataopts)
		opts = allOpts(model_name,
		               netopts=netopts,
		               dataopts=dataopts,
		               optimizeropts=optimopts,
		               epocheropts=epocheropts,
		               )

		return opts

	@abstractmethod
	def collect_opts(self):
		raise NotImplementedError('collect_opts: Must return a list of opts')

	def mean_std_results(self,result_list:List[ResultStruct]) -> Tuple[Dict,Dict,Dict,Dict]:
		'''Calculates the mean var trajectory in the epochs of each scalar value and return a dict similar to the
		   encapsulated dict in ResultStruct'''
		dictlist=[]
		for result in result_list:
			dictlist.append( result.resultdict)
		df = pd.DataFrame(dictlist)
		array = np.array(df.values.tolist(),dtype=np.float64) #type: np.ndarray
		mean_trajectory = array.mean(axis=0)
		lastepochmean = mean_trajectory[0:,-1:]
		std_trajectory = np.std(array,axis=0)
		lastepochstd = std_trajectory[0:,-1:]
		# columns are epochs, This part is empirically test, avoid inferring things
		mean_traj_dict = pd.DataFrame(mean_trajectory.transpose(),columns=df.columns).to_dict('list')
		var_traj_dict = pd.DataFrame(std_trajectory.transpose(),columns=df.columns).to_dict('list')
		lastepochmean = pd.DataFrame(lastepochmean.transpose(),columns=df.columns).to_dict('list')
		lastepochstd = pd.DataFrame(lastepochstd.transpose(), columns=df.columns).to_dict('list')
		return mean_traj_dict, var_traj_dict , lastepochmean,lastepochstd

	def write_summary(self,last_mean_dict_list,last_std_dict_list,optslist:List[allOpts]):
		with xl.Workbook(os.path.join(*[EXP_RESULT_ROOT_DIR,self.name,self.name+'.xlsx'])) as workbook:

			for i,opt in enumerate(optslist):
				dataset_name = opt.dataset_name
				model_name = opt.model_name
				last_mean_dict = last_mean_dict_list[i]
				last_std_dict = last_std_dict_list[i]
				raise NotImplementedError()

	def force_create_path(self,pathlist:List[str]) -> Tuple[str,bool]:
		full_path = ''
		exists = True
		for path in pathlist:
			full_path = os.path.join(full_path,path)
			if not os.path.exists(full_path):
				os.mkdir(full_path)
				exists = False
		return full_path, exists
	def force_replace_path(self,pathlist:List[str]) -> str:
		full_path = ''
		exists = True
		for path in pathlist:
			full_path = os.path.join(full_path, path)
			if not os.path.exists(full_path):
				os.mkdir(full_path)
				exists = False
		if exists:
			os.rmdir(full_path)
			os.mkdir(full_path)
		return full_path
	def validate(self):
		raise NotImplementedError()
	def run(self):
		self.opt_list = self.collect_opts()
		result_dir=[]
		meanpath=[]
		stdpath=[]
		# self.validate()
		for opt in self.opt_list: # type: allOpts
			ep = Epocher(opt)
			opt.print()
			result_list =[]
			last_mean_dict_list= []
			last_std_dict_list = []
			for trial in range(self.trials):
				ep.reinstantiate_model()
				result_dir_list = [self.result_dir,opt.dataopts.datasetname,opt.name,str(trial)]
				if self.save_results:
					result_dir,exists = self.force_create_path(result_dir_list)
					if exists and not self.override:
						raise Exception('Path exists: you did not allow overriding')

				if self.override:
					result_dir, exists = self.force_create_path(result_dir_list)
				print(opt.name +' '+ opt.dataopts.datasetname + ' ' + str(trial))
				result = ep.run_many_epochs(result_dir,self.save_results)
				result_list.append(result)
			mean_traj_dict, std_traj_dict, last_mean_dict, last_std_dict = self.mean_std_results(result_list)
			if self.save_results:
				meanpath, exists_mean = self.force_create_path([self.summary_dir, opt.dataopts.datasetname, opt.name, 'mean'])
				stdpath, exists_std = self.force_create_path([self.summary_dir, opt.dataopts.datasetname, opt.name, 'std'])
				ResultStruct.write_res_dict(mean_traj_dict, meanpath)
				ResultStruct.write_res_dict(std_traj_dict, stdpath)
			last_mean_dict_list.append(last_mean_dict)
			last_std_dict_list.append(last_std_dict)







