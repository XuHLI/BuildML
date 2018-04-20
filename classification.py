import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris

# load data from load_iris
data = load_iris()

features = data.data
feature_name = data.target_names
labels = data.target

# plot the figures 
for label, marker, color in zip(range(3),'>ox','rgb'):
	plt.scatter(features[labels==label,0],features[labels==label,1],
		marker = marker, color=color)
# plt.show()

targets = labels.copy().astype(str)
# print(targets.dtype)
for label in range(3):
	targets[labels==label] = feature_name[label]
# print(targets,featrue_name[0])
is_sentosa = (targets=='setosa')

print('Maximum of setosa: {0}.'.format(features[is_sentosa,2].max()))
print('Minimum of others: {0}.'.format(features[~is_sentosa,2].min()))

# Next find a threshold for virginica. First remove setosa in features and targets
features = features[~is_sentosa]
targets = targets[~is_sentosa]


# sum = 0
# N = targets.shape[0]

# # add cross-validation
# for index in range(N):
# 	instance = features[index,:]
# 	instance_target = targets[index]
# 	label = np.ones(N,bool)

# 	label[index] = False
# 	# print(label)
# 	# break

# 	features_train = features[label]
# 	targets_train = targets[label]

# 	virginica = (targets_train=='virginica')


# 	# find the best threshold for class virginica
# 	best_findex = 0
# 	best_th = 0
# 	best_acc = -1.0
# 	for f_index in range(4):
# 		thresh = features_train[:,f_index].copy()
# 		thresh.sort()
# 		for th in thresh:
# 			pred = features_train[:,f_index]>th
# 			acc = (pred == virginica).mean()
# 			if acc>best_acc:
# 				best_acc = acc
# 				best_findex = f_index
# 				best_th = th

# 	print('Best accuracy: {0}, best feature index: {1}, best threshold: {2}.'.format(
# 		best_acc,best_findex,best_th))

# 	if instance[best_findex]>best_th:
# 		sum += (instance_target=='virginica')
# 	else:
# 		sum += (instance_target==feature_name[1])
		
# print('Cross-validation accuracy: {0}.'.format(sum/N))

# print(feature_name)

def training(feature,label):
	best_acc = -1.0
	virginica = (label=='virginica')
	for f_index in range(4):
		thresh = feature[:,f_index].copy()
		thresh.sort()
		for th in thresh:
			pred = feature[:,f_index]>th
			acc = np.mean(pred == virginica)
			if acc>best_acc:
				best_acc = acc
				best_findex = f_index
				best_th = th
	print('Best accuracy: {0}, best feature index: {1}, best threshold: {2}.'.format(
		best_acc,best_findex,best_th))
	return best_findex, best_th

def accuracy(feature,label,threshold):
	pred = feature>threshold
	data_size = label.shape[0]
	return np.sum(pred==label)/data_size

# k fold cross validation

from sklearn.model_selection import KFold

# split data 
data = [features,targets]
kf = KFold(n_splits=5,random_state=None,shuffle=True)
num_fold = kf.get_n_splits(features,targets)
acc = .0
for train, test in kf.split(features,targets):
	# print(features[train,:],targets[train])

	best_findex, best_th = training(features[train,:],targets[train])
	label = (targets[test]=='virginica')
	acc += accuracy(features[test,best_findex],label,best_th)

print('Average accuracy of {0} fold cross validation is: {1}'.format(
	num_fold,acc/num_fold))








# 	if instance[best_findex]>best_th:
# 		sum += (instance_target=='virginica')
# 	else:
# 		sum += (instance_target==feature_name[1])
		
# print('Cross-validation accuracy: {0}.'.format(sum/N))
