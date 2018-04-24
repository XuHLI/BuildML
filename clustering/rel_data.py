import sklearn.datasets
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import stem
import sys
from sklearn.cluster import KMeans

print(sys.getdefaultencoding())

english_stemmer =  stem.SnowballStemmer('english')
class StemTfidfVectorizer(TfidfVectorizer):
	def bulid_anlayzer(self):
		analyzer = super(StemTfidfVectorizer,self).bulid_anlayzer()
		return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


MLCOMP_DIR='D:\WORK\Python\Building machine learning with python\data'

groups = ['comp.graphics', 'comp.os.ms-windows.misc', 
'comp.sys.ibm.pc.hardware', 'comp.sys.ma c.hardware', 
'comp.windows.x', 'sci.space']

datasets = sklearn.datasets.load_mlcomp("20news-18828", "train",
	mlcomp_root= MLCOMP_DIR,categories=groups)


train_data = datasets.data
# vectorizer using tf-idf method, stemmed, with real data noisy
vectorizer = StemTfidfVectorizer(min_df=10,max_df=0.5,
	stop_words='english', decode_error='ignore')

vectorized = vectorizer.fit_transform(train_data)

num_samples, num_features = vectorized.shape

print("#samples: {0}, #features: {1}".format(num_samples,num_features))

num_clusters = 50
km = KMeans(init='random', n_clusters=num_clusters, verbose=1,n_init=1)
km.fit(vectorized)
# print(km.labels_)
# print(km.cluster_centers_)


new_post = ['Disk drive problems. Hi, \
I have a problem with my hard disk.After 1 year \
it is working only sporadically now. I tried to format \
it, but now it doesn\'t boot any more.Any ideas? Thanks.']

new_post_vec = vectorizer.transform(new_post)

new_label = km.predict(new_post_vec)[0]

similar_indices = (km.labels_ == new_label).nonzero()[0]

similar = []
for i in similar_indices:
	dist = np.linalg.norm((vectorized[i]-new_post_vec).toarray())
	similar.append((dist,train_data[i]))
similar = sorted(similar)
print(similar_indices)
print(len(similar))

print(similar[0])
print(similar[-1])






# preprocessing: learn later
# DATA_DIR = 'D:\WORK\Python\Building machine learning with python\data\original_data'


# target_names = [dir for dir in os.listdir(DATA_DIR)]

# # print(target_names)

# filenames =[]
# for t in target_names:

# 	dir = os.path.join(DATA_DIR,t)
# 	# print(t)
# 	# print(dir)
# 	# for file in os.listdir(dir):
# 	# 	print(file)
# 	# 	break
# 	# filenames.append([os.path.join(dir,file) for file in os.listdir(dir)])
# 	for file in os.listdir(dir):
# 		filenames.append([os.path.join(dir,file)])
# 	# print(filenames)
# 	# break


# filenames = np.array(filenames).flatten()

# print(filenames[1])
# with open(filenames[1],'rb') as file:
# 	reader = file.read()
# 	# print(reader)
# a = open(filenames[1],'rb')
# f = [x.strip() for x in a]	

# print(f)
# # f = open("D:\WORK\Python\Building machine learning with python\data\original_data\\alt.atheism\\51060",'r')

# # test = []
# # for row in f:
# # 	print(row)

# # test = list(open(filenames[0],'rb').read())
# # print(test)