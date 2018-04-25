from gensim import corpora, models, similarities
import os
import numpy as np
import matplotlib.pyplot as plt 
from scipy.spatial import distance 

if os.path.isfile('D:\WORK\Python\Building machine learning with python\BuildML\data\\ap\\vocab.txt'):
	print('Found')
# '/data/ap/vocab.txt'
# 
corpus = corpora.BleiCorpus('D:\WORK\Python\Building machine learning with python\BuildML\data\\ap\\ap.dat',
	'D:\WORK\Python\Building machine learning with python\BuildML\data\\ap\\vocab.txt'
	)

model = models.ldamodel.LdaModel(corpus, num_topics=100,
	id2word=corpus.id2word,alpha=1)

topics = [model[c] for c in corpus]
# print(topics[0])
# for c in corpus:
# 	print(c)
# 	break

## histogram of number of topics vs number of documents
# Nr_topics = np.array([len(topics[t]) for t in range(len(topics))])
# plt.hist(Nr_topics,bins=100,color='r',histtype='stepfilled')
# plt.xlabel('Nr of topics')
# plt.ylabel('Nr of documents')
# plt.autoscale(tight=True)
# plt.text(30,120,r'$\alpha=1.0$')
# plt.grid()
# plt.show()

## compute all pairwise distances of different documents: num_topics=100
dense = np.zeros((len(topics),100),float)
for ti,t in enumerate(topics):
	for tj, tp in t:
		dense[ti,tj] = tp
	
pairwise = distance.squareform(distance.pdist(dense))

# add a larger value to the distance of itself
largest = pairwise.max()

for ti in range(len(topics)):
	pairwise[ti,ti] = largest+1


# find the closest document for each document
def closest_to(doc_id):
	return pairwise[doc_id].argmin()

print(closest_to(1))