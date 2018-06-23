import csv
import collections
import operator
import pickle
import os
import numpy as np


### Load data

track_to_vector_dict = {}
track_to_genre_dict = {}
genre_to_tracks_dict = {}

with open('mxm_track_to_vector_dict.pkl', 'rb') as f: 
	track_to_vector_dict = pickle.load(f)

with open('mxm_track_to_genre_dict.pkl', 'rb') as f: 
	track_to_genre_dict = pickle.load(f)

with open('genre_to_mxm_track_dict.pkl', 'rb') as f: 
	genre_to_tracks_dict = pickle.load(f)


### Get genre mean vectors

genre_mean_dict = {}

for genre in genre_to_tracks_dict:
	tracks = genre_to_tracks_dict[genre]
	vecs = []
	for track in tracks: 
		if track in track_to_vector_dict.keys():
			vecs.append(track_to_vector_dict[track])
	genre_mean_dict[genre] = np.mean(vecs, axis=0)


### Train/test split

overlap = []
for key in track_to_genre_dict.keys():
	if key in track_to_vector_dict.keys():
		overlap.append(key)

genre_list = []
track_list = []
for o in overlap:
	genre_list.append(track_to_genre_dict[o])

#x = [track_to_vector_dict for o in overlap] # for unclean version
#y = [track_to_genre_dict for o in overlap] # for unclean version

x = []
y = []
count_dict = collections.defaultdict(int)

for o in overlap:
	if track_to_genre_dict[o] != 'r&b':
		if count_dict[track_to_genre_dict[o]] < 4216:
			count_dict[track_to_genre_dict[o]] += 1
			x.append(track_to_vector_dict[o])
			y.append(track_to_genre_dict[o])

import random

c = list(zip(x, y))
random.shuffle(c)
x, y = zip(*c)
x = list(x)
y = list(y)

x_train = x[:-3000]
y_train = y[:-3000]
x_test = x[-3000:]
y_test = y[-3000:]


### Create new feature: cosine distance from genre mean vector

from scipy.spatial.distance import cosine

def get_distance_feature(vectors):
	result = []
	for vec in vectors:
		features = []
		for genre in genre_mean_dict.keys():
			features.append(cosine(genre_mean_dict[genre], vec))
		result.append(features)
	return result

x_train_dist = get_distance_feature(x_train)
x_test_dist = get_distance_feature(x_test)

### Train classifier

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB 
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix


#clf = MultinomialNB()

pipeline = Pipeline([
    ('tfidf_transformer', TfidfTransformer()),
    ('classifier', MultinomialNB())
    #('classifier', GaussianNB())
    #('classifier', SGDClassifier())
])

clf = LogisticRegression()

#clf.fit(x_train, y_train)
#pipeline.fit(x_train, y_train)
#clf.fit(x_train_dist, y_train)

### Test classifier 

y_pred = clf.predict(x_test)
#y_pred = pipeline.predict(x_test)
#y_pred = clf.predict(x_test_dist)

acc = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(list(y_test), list(y_pred))

print(acc)

#print(list(clf.classes_))
print(confusion)


