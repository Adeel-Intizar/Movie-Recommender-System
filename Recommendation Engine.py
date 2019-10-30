#!/usr/bin/env python
# coding: utf-8

# ## Importing Essential Libraries

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ## Functions to get Info

def get_title(index):
	return df[df.index == index]["title"].values[0]

def get_index(title):
	return df[df.title == title]["index"].values[0]


df = pd.read_csv('movie_dataset.csv')
df.columns


# ## Selecting Features

features = df[['keywords', 'genres', 'cast', 'director', 'title', 'tagline']]


features.head()


for i in features:
    features[i] = features[i].fillna('')


def combine(row):
    return row['keywords']+ " "+row['genres']+" "+row['cast']+" "+row['director'] + " " +row['title'] + " " +row['tagline']


features['combined_features'] = features.apply(combine, axis = 1)
print(features['combined_features'].head())


# ## Applying Cosine Similarity


cv = CountVectorizer()
cm = cv.fit_transform(features['combined_features'])


cosine_similarity = cosine_similarity(cm)
user = 'Avatar'


index = get_index(user)



# ## Getting Top 5 Recommended Movies


print('Movie Recommender System \n')
user = input('Enter Movie Name: ')
print('\nRecommended Movies:')
index = get_index(user)
similar = list(enumerate(cosine_similarity[index]))
sorted_movies = sorted(similar, key = lambda x: x[1], reverse = True)
for i, movie in enumerate(sorted_movies):
    print('\n \t',get_title(movie[0]))
    if i > 5:
        break
