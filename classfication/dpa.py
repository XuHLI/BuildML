import os
import json

import re
import numpy as np

import matplotlib.pyplot as plt 
from sklearn import neighbors
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, classification_report

DATA_DIR = 'D:\WORK\Python\Building machine learning with python\data\classfication'

filename = os.path.join(DATA_DIR,'chosen.tsv')
filename_meta = os.path.join(DATA_DIR,'chosen-meta.json')

# def load_meta(filename):
#     meta = json.load(open(filename, "r"))
#     keys = list(meta.keys())

#     # JSON only allows string keys, changing that to int
#     for key in keys:
#         meta[int(key)] = meta[key]
#         del meta[key]

#     # post Id to index in vectorized
#     id_to_idx = {}
#     # and back
#     idx_to_id = {}

#     for PostId, Info in meta.items():
#         id_to_idx[PostId] = idx = Info['idx']
#         idx_to_id[idx] = PostId

#     return meta, id_to_idx, idx_to_id



def fetch_post():
	for line in open(filename,'r'):
		post_id, text = line.split('\t')
		yield int(post_id), text.strip()

chosen_meta = json.load(open(filename_meta,'r'))

all_answers = [q for q, v in chosen_meta.items() if v['ParentId'] !=-1]


Y = np.asarray([chosen_meta[aid]['Score']>0 for aid in all_answers])



code_match = re.compile('<pre>(.*?)</pre>',re.MULTILINE|re.DOTALL)
link_match = re.compile('<a href="http://.*?".*?>(.*?)</a>',re.MULTILINE|re.DOTALL)
img_match = re.compile('<img(.*?)/>', re.MULTILINE | re.DOTALL)
tag_match = re.compile('<[^>]*>', re.MULTILINE | re.DOTALL)

def extract_features_from_body(s):
	num_code_lines = 0
	link_count_in_code = 0
	code_free_s = s
	for match_str in code_match.findall(s):
		num_code_lines += match_str.count('\n')
		code_free_s = code_match.sub("",code_free_s)
		
		link_count_in_code+= len(link_match.findall(match_str))
		
	links = link_match.findall(s)
	link_count = len(links)
	link_count -= link_count_in_code

	html_free_s = re.sub(" +", " ", tag_match.sub('', 
		code_free_s)).replace("\n","")
	link_free_s = html_free_s

	for link in links:
		if link.lower().startswith("http://"):
			link_free_s = link_free_s.replace(link,'')
	num_text_tokens = html_free_s.count(" ")

	return num_text_tokens, num_code_lines, link_count

def lr_model(clf,X):
	return 1/(1+np.exp(-(clf.intercept_ + np.sum(clf.coef_ *X))))




	# return len(link_match.findall(s)) - link_count_in_code

X = np.asarray([extract_features_from_body(text) for post_id, text in 
	fetch_post() if str(post_id) in all_answers])

# plt.hist(X[:,1],bins=100)
# plt.autoscale(tight=True)
# plt.show()
# print(X.shape)

# clf = LogisticRegression()
# print(clf)
# clf.fit(X,Y)

# print("The probability of a test result is {0}".format(
# 	lr_model(clf,[1,2,3])))


# model KNN and Logistic Regression
scores = []
precisions = []
recalls = []
thresholds = []
pre_scores = []

cv = KFold(n=len(X),n_folds=10,shuffle=True)
for train,test in cv:
	X_train, y_train = X[train], Y[train]
	X_test, y_test = X[test], Y[test]
	# KNN
	# clf = neighbors.KNeighborsClassifier()
	# clf.fit(X_train,y_train)
	# scores.append(clf.score(X_test,y_test))

	# Logistic Regression
	clf = LogisticRegression(C=0.1)
	clf.fit(X,Y)
	scores.append(clf.score(X_test,y_test))
	proba = clf.predict_proba(X_test)
	# print(proba)
	precision, recall, threshold = precision_recall_curve(y_test,proba[:,1])
	precisions.append(precision)
	recalls.append(recall)
	thresholds.append(threshold)
	# print(threshold)

	pre_scores.append(auc(recall,precision))
	print(classification_report(y_test,proba[:,1]>0.59))
	# print(classification_report(y_test,proba[:,1]>0.59,target_names=['not accepted',
	# 	'accepted']))

	# thresholds = np.hstack(([0],thresholds))
	# idx80 = precsion>=0.8
	# print('P={0} R={1}, thresh = {2}'.format(precsion,
	# 	recall,thresholds))
	# print('P={0} R={1}, thresh = {2}'.format(precsion[idx80][0],
	# 	recall[idx80][0],thresholds[idx80][0]))

	# plt.step(recall,precsion)
	# plt.show()

scores_to_sort = pre_scores
# print(scores_to_sort, len(scores_to_sort)/2)
medium = np.argsort(scores_to_sort)[int(len(scores_to_sort)/2)]

threshold_ = np.hstack(([0],thresholds[medium]))
# print(threshold_)
precisions = precisions[medium]
recalls = recalls[medium]

idx80 = precisions >= 0.8
print('P={0} R={1}, thresh = {2}'.format(precisions[idx80][0],
	recalls[idx80][0],threshold_[idx80][0]))



print("Mean(scores) = {0} \t Stddev(scores)={1}".format(np.mean(scores),
	np.std(scores)))
