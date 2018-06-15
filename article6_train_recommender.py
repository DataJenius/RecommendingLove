#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train our NNMF recommender and save resulting dataframes to pickles
"""

################################################################################
# load required dependencies
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import stop_words
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

################################################################################
# load our data and clean it up a bit
startTime = datetime.now()
df = pd.read_csv("data/profiles.csv")
df = df.fillna('')

# organize into a cleaner dataframe
df["essay"] = df["essay0"]
df = pd.DataFrame(data={'id': 0, 
                        'age': df['age'],
                        'sex': df['sex'],
                        'orientation': df['orientation'],
                        'essay': df['essay']})
df = df[['id','age','sex','orientation','essay']]
df = df.loc[(df["essay"] != '')]
df = df.reset_index(drop=True)
df["id"] = df.index
print(df.head())
print("------------------------------------------------")
print("Imported Data in ",datetime.now() - startTime)
print("------------------------------------------------")



################################################################################
# Create tfidf matrix from text input 
startTime = datetime.now()
sw = list(stop_words.ENGLISH_STOP_WORDS)
sw.extend(['br','em','strong','class','ilink','href','don','ve','http','https','www','youtube','watch','target','nofollow','rel','amp','san','francisco','bay','area','new','people','like','things'])
tfidf = TfidfVectorizer(stop_words=sw,ngram_range=(2,2),min_df=10) 
tfidf_mat = tfidf.fit_transform(df["essay"])
terms = tfidf.get_feature_names()
print("Created tf-idf matrix in ",datetime.now() - startTime)
print("------------------------------------------------")



################################################################################
# Import NMF and fit our NMF model
startTime = datetime.now()
model = NMF(n_components=20)
model.fit(tfidf_mat)
nmf_features = normalize(model.transform(tfidf_mat))
print("Fit NMF model in ",datetime.now() - startTime)
print("------------------------------------------------")


################################################################################
# Create a dataframe with NMF features indexed to user id
startTime = datetime.now()
df_NMF = pd.DataFrame(nmf_features, index=df["id"])

# create a dataframe with terms indexed to components
df_components = pd.DataFrame(model.components_, columns=terms)

# save dataframes as pickles
df_NMF.to_pickle('data/df_NMF.pkl')
df_components.to_pickle('data/df_components.pkl')
df = df.to_pickle('data/df_users.pkl')
print("Saved pickle files in ",datetime.now() - startTime)
print("------------------------------------------------")

