import os
import csv
import json
import numpy as np 



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


			

X, Y = load_sanders_data()

classes = np.unique(Y)

for c in classes:
	print("#%s: %i" %(c, sum(Y==c)))
