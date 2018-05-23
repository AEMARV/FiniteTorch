from parser.staticnets import StaticNet
def get_model_module(model_name,opts):
	model_string = locals()[model_name]()
	module = StaticNet(model_string, opts)
	return module,opts