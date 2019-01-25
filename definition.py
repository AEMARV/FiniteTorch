import os
import torch
PATH_ROOT = os.path.dirname(os.path.abspath(__file__))
PATH_DATA = os.path.join(PATH_ROOT,'data','datasets')


# OPTS NAMES
OPTS_DATASET = 'dataset'
OPTS_DATASET_CHANS = 'datachandim'
OPTS_DATASET_SPATIAL = 'datarcdim'
OPTS_OPTMIZER = 'optimizer'
OPTS_MODEL = 'model'
epsilon = 1e-7

#Result defs
RESULT_ROOT_DIR = './results'
EXP_RESULT_ROOT_DIR = './experiment_conclude'
def hasnan(t):
	if torch.isnan(t).sum()>0:
		return True
	return False
def hasinf(t):
	if (t == float('inf')).sum() >0:
		return True
	return False