from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os
import numpy as np
import sys
import nltk.stem # natural language processing toolkit
import math

# vectorizer = CountVectorizer(min_df=1)

# # print(vectorizer)

# content = ['How to format my hard disk', 'Hard disk format problems']

# X = vectorizer.fit_transform(content)

# print(vectorizer.get_feature_names(),X.toarray().transpose())

# play with toy

# similarity measurement: based on word count only
def dist_raw(v1,v2):
	dist = v1-v2
	return np.linalg.norm(dist.toarray())

# similarity measurement: based on normalized word count
def dist_norm(v1,v2):
	v1 = v1/np.linalg.norm(v1.toarray())
	v2 = v2/np.linalg.norm(v2.toarray())
	return np.linalg.norm((v1-v2).toarray())

# Term frequency - inverse document frequency (TF-IDF)
def tf_idf(term, doc, docset):
	print(sum(w.count(term) for w in docset))
	tf = float(doc.count(term)/sum(w.count(term) for w in docset))
	idf = math.log(float(len(docset))/len([doc for doc in docset 
		if term in doc]))
	return tf*idf




# extend vectorizer with NLTK's stemmer
english_stemmer = nltk.stem.SnowballStemmer('english')
class StemCountVectorizer(CountVectorizer):
	def build_analyzer(self):
		analyzer = super(StemCountVectorizer,self).build_analyzer()
		return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

# extend tfidf with NLTK's stemmer
class StemTfidfVectorizer(TfidfVectorizer):
	def build_analyzer(self):
		analyzer = super(StemTfidfVectorizer,self).build_analyzer()
		return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

posts = [open(os.path.join('data/toy',f)).read() for f in 
os.listdir('data/toy')] 

# removing less important words: stop words
# vectorizer = CountVectorizer(min_df=1,stop_words='english')
# vectorizer = CountVectorizer(min_df=1)
# vectorizer = StemCountVectorizer(min_df=1,stop_words='english')
vectorizer = StemTfidfVectorizer(min_df=1,stop_words='english',
	decode_error='ignore')

X_train = vectorizer.fit_transform(posts)

num_samples, num_features = X_train.shape

print(vectorizer.get_feature_names())
# print(X_train.toarray().transpose())

best_doc = None
best_dist = sys.maxsize
best_i = None
new_post = ['imaging databases']
new_post_vec = vectorizer.transform(new_post)
print(new_post_vec.toarray())


for i in range(num_samples):
	post = posts[i]

	if post == new_post:
		continue
	# post_vec = vectorizer.transform([post])
	post_vec = X_train.getrow(i)
	# dist = dist_raw(new_post_vec,post_vec)
	dist = dist_norm(new_post_vec,post_vec)
	print('=== post {0} with dist = {1}: {2}'.format(i,dist,post))

	if dist < best_dist:
		best_dist = dist
		best_i = i

print('Best post is: {0} with dist = {1}'.format(best_i,best_dist))

# stemming
s = nltk.stem.SnowballStemmer('english')
print(s.stem('imagining'))

# doc1, doc2, doc3 = ["a"],['a','b','b'], ['a','b','c']
# D = [doc1,doc2,doc3]
# print(tf_idf('b',doc2,D))
# print(tf_idf('b',doc3,D))
