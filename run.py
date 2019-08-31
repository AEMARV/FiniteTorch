from models.modelutils import *
from data.datasetutils import *
from trainvalid.optimizer import *
from trainvalid.epocher import *
from experiment.Experiments import *

def boolprompt(question):
	answer=''
	while(answer.lower!='n' or answer.lower()!='y'):
		answer = 'n'# input(question+' [y]/[n]')
		if answer[0].lower()=='y':
			return True
		elif answer[0].lower()=='n':
			return False
		else:
			print('please answer with y/n characters')



if __name__ == '__main__':
	exp = MAP(1)
	# exp = Synthetic_PMaps(1)
	exp.run()

