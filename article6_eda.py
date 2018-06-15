#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis of the OKCupid profile data   

Looking at the 54,458 profiles with non-empty essay0
sex = 'm', 'f'
orientation = 'straight', 'bisexual', 'gay'
age is continuous from 18 to 69
"""
################################################################################
# load required dependencies
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns


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
# plot bloxplot of male and female ages
mpl.style.use('seaborn')
mpl.rcParams.update({'font.size': 100})
my_pal = {"f": "#f97a89", "m": "#4286f4"}
my_boxplot = sns.boxplot(x='sex', y='age', data=df, palette=my_pal)
fig = my_boxplot.get_figure()
fig.set_size_inches(11, 8)
plt.title("Ages of Profiles", loc="center")
plt.ylabel("Age")
plt.xlabel("Male v. Female")
fig.savefig("jojo.png") 


################################################################################
# get top X ngrams from a given sub-group of users
startTime = datetime.now()
#group = df.loc[(df["age_group"] == '18-26') & (df["orientation"]=='straight') & (df["sex"]=='m')]
#group = df.loc[(df["age_group"] == '18-26') & (df["orientation"]=='bisexual') & (df["sex"]=='m')]
#group = df.loc[(df["age_group"] == '18-26') & (df["orientation"]=='gay') & (df["sex"]=='m')]
#group = df.loc[(df["age_group"] == '27-34') & (df["orientation"]=='straight') & (df["sex"]=='m')]
#group = df.loc[(df["age_group"] == '35-69') & (df["orientation"]=='straight') & (df["sex"]=='m')]
#group = df.loc[(df["age_group"] == '35-69') & (df["orientation"]=='straight') & (df["sex"]=='f')]
group = df.loc[(df["orientation"]=='gay')]

################################################################################
# use CountVectorizer to get basic word counts of the corpus
# we want most common words for men and women

# Create tfidf matrix from text input 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import stop_words


# handle our stop words
sw = list(stop_words.ENGLISH_STOP_WORDS)
sw.extend(['br','em','strong','class','ilink','href','don','ve','http','https','www','youtube','watch','target','nofollow','rel','amp','_blank','bay','area','san','francisco','like','things','new','meet','sf'])

# run count vectorizer -- # cv.vocabulary_
# we've got ~50k words and ~ 792k bigrams
cv = CountVectorizer(stop_words=sw,ngram_range=(2,2))
cv_fit=cv.fit_transform(group["essay"])
words = cv.get_feature_names()
term_freqs = np.asarray(cv_fit.sum(axis=0)).ravel()

# build dataframe to understand results
df_tf = pd.DataFrame(data={'n': term_freqs,'word': words})
df_tf = df_tf.sort_values(by='n', ascending=False)
print(df_tf.head(n=20))
print("------------------------------------------------")
print("Generated ngrams in ",datetime.now() - startTime, len(group), 'users')
print("------------------------------------------------")
