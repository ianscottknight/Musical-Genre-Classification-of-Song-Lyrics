import csv
import collections
import operator
import pickle
import os

def convert_to_ndarray(d):
    arr = [0] * 5000
    for key, val in d.items():
        key = int(key) - 1
        arr[key] = val
    return arr

track_to_vector_dict = {}

with open('mxm_dataset_train.txt', 'r') as f:
	out = f.readlines()
	for line in out: 
		line = line.strip('\n')
		if line[0] != '#': 
			if line[0] == '%': 
				VOCAB = line.split(',')
			else: 
				d = {}
				splitted = line.split(',')
				_ = splitted[0]
				tid = splitted[1]
				for unsplit_dict in splitted[2:]:
					pair = unsplit_dict.split(':')
					index = int(pair[0])
					count = int(pair[1])
					d[index] = count
				track_to_vector_dict[tid] = convert_to_ndarray(d)

with open('mxm_track_to_vector_dict.pkl', 'wb') as f: 
	pickle.dump(track_to_vector_dict, f)