import os
import csv
import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import precision_recall_curve, auc, f1_score
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.base import BaseEstimator

import nltk

import numpy as np 

import csv, collections



DATA_DIR = 'D:\WORK\Python\Building machine learning with python\data\classification1\data'

def load_sanders_data(dirname='.',line_count=-1):
	count = 0

	topics = []
	labels = []
	tweets = []

	with open(os.path.join(DATA_DIR,dirname,'corpus.csv'),'r') as csvfile:
		metareader = csv.reader(csvfile,delimiter=',',quotechar='"')
		for line in metareader:
			# print(line)
			
			
			count += 1
			
			if line_count > 0 and count > line_count:
				break
			
			topic, label, tweet_id = line
			# print(tweet_id)

			tweet_fn = os.path.join(
				DATA_DIR,dirname,'rawdata', '%s.json'%tweet_id)
			# print(tweet_fn)

			try:
				file = open(tweet_fn,"r")
				tweet = json.load(file)
			except IOError:
				print("Tweet '%s' not found. Skip."% tweet_fn)
				continue

			# print(tweet)
			# break

			if 'text' in tweet and tweet['user']['lang'] == 'en':
				topics.append(topic)
				labels.append(label)
				tweets.append(tweet['text'])
	
	tweets = np.asarray(tweets)
	labels = np.asarray(labels)

	return tweets, labels

# In this way, pipeline can use fit and predict
def create_ngram_model():
	# tfidf_ngrams = TfidfVectorizer(ngram_range=(1,3), analyzer='word',
	# 	binary=False)
	def preprocessor(tweet):
		global emoticons_replaced
		tweet = tweet.lower()

		for k in emo_repl_order:
			tweet = tweet.replace(k,emo_repl[k])

		# print(re_repl.items())

		for r, repl in re_repl.items():
			tweet = re.sub(r,repl,tweet)

		return tweet.replace("-"," ").replace("_"," ")




	tfidf_ngrams = TfidfVectorizer(preprocessor = preprocessor, ngram_range=(1,2),analyzer='word',
		binary=True, smooth_idf=False)
	# tfidf_ngrams = TfidfVectorizer(ngram_range=(1,2),analyzer='word',
	# 	binary=True, smooth_idf=False)
	ling_stats = LinguisticVectorizer()
	all_features = FeatureUnion([('ling',ling_stats),('tfidf',tfidf_ngrams)])
	clf = MultinomialNB()

	pipeline = Pipeline([('all', all_features),('clf',clf)])
	# pipeline = Pipeline([('vect', tfidf_ngrams),('clf',clf)])
	return pipeline

def train_model(clf_factory, X,Y):
	# setting random_state to get deterministic behavior
	cv = ShuffleSplit(n=len(X),n_iter=10,test_size=0.3, random_state=0)

	scores = []
	pre_scores = [] # store area under curve

	for train, test in cv:
		X_train, y_train = X[train],  Y[train]
		X_test, y_test = X[test], Y[test]

		clf = clf_factory()
		clf.fit(X_train,y_train)

		train_score = clf.score(X_train,y_train)
		test_score = clf.score(X_test,y_test)

		scores.append(test_score)

		proba = clf.predict_proba(X_test)

		precision, recall, threshold = precision_recall_curve(y_test,
			proba[:,1])

		pre_scores.append(auc(recall,precision))

	print("Mean(scores): %.3f\nStd(scores):%.3f\nMean(pre_scores):%.3f\nStd(pre_score):%.3f"%(np.mean(scores),
		np.std(scores),np.mean(pre_scores), np.std(pre_scores)))
			
def tweak_labels(Y,pos_sent_list):
	pos = Y==pos_sent_list[0]
	# print(pos)

	for sent_label in pos_sent_list[1:]:
		pos |= Y==sent_label
	

	Y = np.zeros(Y.shape[0])
	Y[pos] = 1
	Y = Y.astype(int)
	# print(Y)


	return Y

def grid_search_model(clf_factory,X,Y):
	cv = ShuffleSplit(n=len(X),n_iter=10,
		test_size=0.3, random_state=0)

	# param_grid = {'vect_ngram_range':[(1,1),(1,2),(1,3)],
		# 'vect_stop_words':[None,"english"]}
	# param_grid = 
	param_grid = dict(vect__ngram_range =[(1,1),(1,2),(1,3)],
		vect__stop_words=[None, "english"],
		vect__smooth_idf=[False,True],
		vect__sublinear_tf=[False,True],
		vect__use_idf=[False,True],
		vect__binary=[False,True],
		vect__min_df=[1,2],
		clf__alpha=[0.01,0.05,0.1,0.5,1])

	grid_search = GridSearchCV(clf_factory(),
		param_grid=param_grid,
		cv=cv,
		verbose=10)

	grid_search.fit(X,Y)

	return grid_search.best_estimator_

# load sentiment words
def load_sent_word_net():

	sent_scores = collections.defaultdict(list)

	with open(os.path.join(DATA_DIR,'SentiWordNet_3.0.0_20130122.txt'),'r') as csvfile:
		reader = csv.reader(csvfile,delimiter='\t',quotechar='"')

		for line in reader:
			if line[0].startswith("#"):
				continue
			if len(line) == 1:
				continue

			POS, ID, PosScore, NegScore, SynsetTerms, Gloss = line
			if len(POS) == 0 or len(ID) == 0:
				continue

			for term in SynsetTerms.split(' '):

				term = term.split("#")[0]
				# print(term)
				term = term.replace("-", " ").replace("_"," ")
				# print(term)
				key = "%s/%s"%(POS, term.split("#")[0])
				sent_scores[key].append((float(PosScore),
					float(NegScore)))

				

	for key, value in sent_scores.items():
		sent_scores[key] = np.mean(value, axis=0)

	return sent_scores

class LinguisticVectorizer(BaseEstimator):
	def get_feature_names(self):
		return np.array(['sent_neut', 'sent_pos', 'sent_neg', 'nouns', 'adjectives', 'verbs', 'adverbs',
			'allcaps', 'exclamation', 'question', 'hashtag','mentioning'])

	def fit(self,documents,y=None):
		return self

	def _get_sentiments(self,d):
		sent = tuple(d.split())

		tagged = nltk.pos_tag(sent)

		pos_vals = []
		neg_vals = []

		nouns = 0
		adjectives = 0
		verbs = 0
		adverbs = 0

		for w, t in tagged:
			p, n = 0,0
			sent_pos_type = None
			if t.startswith("NN"):
				sent_pos_type = "n"
				nouns += 1
			elif t.startswith("JJ"):
				sent_pos_type = "a"
				adjectives += 1
			elif t.startswith("VB"):
				sent_pos_type = "v"
				verbs += 1
			elif t.startswith("RB"):
				sent_pos_type = "r"
				adverbs += 1

			if sent_pos_type is not None:
				sent_word = "%s/%s"%(sent_pos_type,w)

				if sent_word in sent_word_net:
					p, n = sent_word_net[sent_word]

			pos_vals.append(p)
			neg_vals.append(n)

		l = len(sent)
		avg_pos_val = np.mean(pos_vals)
		avg_neg_val = np.mean(neg_vals)
		return [1-avg_pos_val-avg_neg_val, avg_pos_val, avg_neg_val,
		nouns/l, adjectives/l,verbs/l, adverbs/l]

	def transform(self,documents):
		obj_val, pos_val, neg_val, nouns, adjectives,\
		 verbs,adverbs = np.array([self._get_sentiments(d) for d in documents]).T 

		allcaps = [] 
		exclamation = []
		question = []
		hashtag = []
		mentioning = []

		for d in documents:
			allcaps.append(np.sum([t.isupper() for t in d.split() if len(t)>2]))

			exclamation.append(d.count("!"))
			question.append(d.count("?"))
			hashtag.append(d.count("#"))
			mentioning.append(d.count("@"))

		result = np.array([obj_val, pos_val, neg_val,nouns, adjectives, verbs, adverbs,
			allcaps, exclamation, question,hashtag, mentioning]).T

		return result






X_org, Y_org = load_sanders_data()

classes = np.unique(Y_org)

for c in classes:
	print("#%s: %i" %(c, sum(Y_org==c)))


## define a range of frequent emoticons and their replacements
emo_repl = {
# positive emoticons
"&lt;3": " good ",
":d": " good ", # :D in lower case
":dd": " good ", # :DD in lower case
"8)": " good ",
":-)": " good ",
":)": " good ",
";)": " good ",
"(-:": " good ",
"(:": " good ",
# negative emoticons:
":/": " bad ",
":&gt;": " sad ",
":')": " sad ",
":-(": " bad ",
":(": " bad ",
":S": " bad ",
":-S": " bad ",
}

emo_repl_order = [k for (k_len,k) in reversed(sorted(
	[(len(k),k)for k in emo_repl.keys()] ))]

print(emo_repl_order)

# define abbreviations as regular expressions
re_repl = {
r"\br\b": "are",
r"\bu\b": "you",
r"\bhaha\b": "ha",
r"\bhahaha\b": "ha",
r"\bdon't\b": "do not",
r"\bdoesn't\b": "does not",
r"\bdidn't\b": "did not",
r"\bhasn't\b": "has not",
r"\bhaven't\b": "have not",
r"\bhadn't\b": "had not",
r"\bwon't\b": "will not",
r"\bwouldn't\b": "would not",
r"\bcan't\b": "can not",
r"\bcannot\b": "can not",
}


# only consider positive and negative examples
# pos_neg_idx = np.logical_or(Y_org=='positive', Y_org=='negative')

# X = X_org[pos_neg_idx]
# Y = Y_org[pos_neg_idx]

# positive：labeled by True, negative: labeled by False
# Y = Y=='positive'
# pos_sent_list = ["positive","negative"]
pos_sent_list = ["positive"]
# pos_sent_list = ["negative"]
Y = tweak_labels(Y_org,pos_sent_list)

# train_model(create_ngram_model,X_org,Y)
# print(grid_search_model(create_ngram_model,X_org,Y))

# determining the word types: part of speech
# import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# print(nltk.pos_tag(nltk.word_tokenize('This is a good book.')))


sent_word_net = load_sent_word_net()
train_model(create_ngram_model,X_org,Y)