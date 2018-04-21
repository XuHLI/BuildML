import numpy as np 
import pickle
import scipy as sp 
import pandas as pd
import csv
import os

# load data 
# data = pd.read_table('data/seeds.tsv',delimiter='\t')

# data = data.as_matrix
# print(data)

# lines = [line.rstrip() for line in open('data/seeds.tsv')]

# for i in range(len(lines)):
# 	value = re.split()

### preprocessing
if os.path.exists('data_seed.pickle'):

	print('Found!!! data loading')
	features = pickle.load(open('data_seed.pickle','rb'))
	labels = pickle.load(open('label_seed.pickle','rb'))
else:
	print('Not Found!!! Preprocessing first and save')
	data = []
	with open('data/seeds.tsv') as tsvfile:
		reader = csv.reader(tsvfile,delimiter='\t')
		for row in reader:
			
			data.append(row)

	# convert to np.array so that we can familiar techniques		
	data = np.array(data)
	features = data[:,:-1].astype('float') # convert "str" to "float"
	labels = data[:,-1] # string
	# print(features[1,:])
	# features = np.array(data[:,:-1])

	#save data
	file1 = open('data_seed.pickle','wb')
	pickle.dump(features,file1)
	file1.close()
	file2 = open('label_seed.pickle','wb')
	pickle.dump(labels,file2)
	file2.close()

# define similarity 
def distance(x,y):
	return np.dot(x-y,x-y)

# classifier
def nn_classify(training_set,training_label,newexample):
	dist = np.array([distance(t,newexample) for t in training_set])

	nearest = dist.argmin()

	return training_label[nearest]

label = nn_classify(features[1:10],labels[1:10],features[15])

print(label)

# normalization of the features: Z-score
features -= features.mean(axis=0)
features /= features.std(axis=0)

## K-fold cross validation
from sklearn.model_selection import KFold 

kf = KFold(n_splits=6,random_state=0,shuffle=True)
num_fold = kf.get_n_splits(features)

acc = 0
for train, test in kf.split(features,labels):
	label_classify = np.array([nn_classify(features[train],labels[train],ex)
		for ex in features[test]])
	label_test = labels[test]
	acc += np.mean(label_classify==label_test)

print('{0} fold cross validation accuracy is: {1}'.format(num_fold,acc/num_fold))









