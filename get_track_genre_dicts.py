import csv
import collections
import operator
import pickle
import os

### PHASE 1 ###

GENRES = ['hip-hop', 'country', 'rock', 'electronic', 'pop', 'r&b', 'metal', 'alternative']

tags_dict = {}

with open('tags.csv', 'r') as f: 
	reader = csv.reader(f)
	next(reader, None) ### skip header
	for i, row in enumerate(reader): 
		tags_dict[i+1] = row[0].lower() ### lowercase only

tracks_dict = {}
track_to_scores_dict = collections.defaultdict(list)

with open('tids.csv', 'r') as f: 
	reader = csv.reader(f)
	for i, row in enumerate(reader): 
		tracks_dict[i+1] = row[0]
		track_to_scores_dict[i+1] = collections.defaultdict(int)

with open('tid_tag.csv', 'r') as f: 
	reader = csv.reader(f)
	for i, row in enumerate(reader): 
		for genre in GENRES: 
			track_id = int(row[0])
			tag_id = int(row[1])
			score = float(row[2])
			if genre in tags_dict[tag_id]:
				track_to_scores_dict[track_id][genre] += score
			else:
				track_to_scores_dict[track_id][genre] += 0.0

track_to_genre_dict = {}
genre_to_tracks_dict = collections.defaultdict(list)

for track_id in track_to_scores_dict.keys(): 
	track = tracks_dict[track_id]
	highest_scoring_genre = max(track_to_scores_dict[track_id].items(), key=operator.itemgetter(1))[0]
	track_to_genre_dict[track] = highest_scoring_genre
	genre_to_tracks_dict[highest_scoring_genre] += [track]

counts = {}
for genre in GENRES: 
	counts[genre] = len(genre_to_tracks_dict[genre])

### PHASE 2 ###
 
sep = '<SEP>'
mxm_track_to_genre_dict = {}
track_to_mxm_track_dict = {}
with open('mxm_779k_matches.txt', 'r') as f:
	out = f.readlines()
	for line in out: 
		if line[0] != '#': 
			elements = line.split(sep)
			old_id = elements[0]
			new_id = elements[3]
			try:
				mxm_track_to_genre_dict[new_id] = track_to_genre_dict[old_id]
				track_to_mxm_track_dict[old_id] = new_id
			except: 
				pass

with open('mxm_track_to_genre_dict.pkl', 'wb') as f: 
	pickle.dump(mxm_track_to_genre_dict, f)

genre_to_mxm_track_dict = collections.defaultdict(list)

for genre in genre_to_tracks_dict.keys():
	for track in genre_to_tracks_dict[genre]:
		try: 
			new_track_id = track_to_mxm_track_dict[track]
			genre_to_mxm_track_dict[genre] += [new_track_id]
		except:
			pass

with open('genre_to_mxm_track_dict.pkl', 'wb') as f: 
	pickle.dump(genre_to_mxm_track_dict, f)




