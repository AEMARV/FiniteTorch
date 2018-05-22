from parser import staticnets
def get_model_module(model_name,opts):
	model_string,opts = locals()[model_name](opts)
	module = staticnets(model_string, opts)
	return module,opts