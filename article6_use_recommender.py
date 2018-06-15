#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Use our recommender to find matches for a given user

Some fun bigrams to look at:
    
- appreciate art
- zombie movies
- rock climbing
- berkeley grad

Use this code to find users witha specific text:
#sub = df_users[df_users['essay'].str.contains("berkeley grad")]

get specific NMF vector for user
df_NMF.iloc[31700]

get bigrams of a given component (0-20)
df_components.iloc[0].nlargest(n=100)

These are the matches used in the article

#3971   31   f    straight  "zombie movies" -- 42531 -- share component 0 
#31700  27   m    straight  "appreciate art" -- 4313 -- share component 15
#7213   24   m    gay       "law school" -- 13418 -- share component 8

    
"""
################################################################################
# we will make matches for the user id defined here
our_user_id = 3971

################################################################################
# load required dependencies
import pandas as pd
from datetime import datetime

# load our pickle files
startTime = datetime.now()
df_components = pd.read_pickle('data/df_components.pkl')
df_NMF = pd.read_pickle('data/df_NMF.pkl')
df_users = pd.read_pickle('data/df_users.pkl')
print("------------------------------------------------")
print("Loaded pkl files in ",datetime.now() - startTime)
print("------------------------------------------------")



# target a specific user
startTime = datetime.now()
main_feature_id  = df_NMF.iloc[our_user_id].nlargest(n=1).index[0]  # the main NMF feature assigned to this user
print("----------------------------------")
print("OUR SELECTED USER:",our_user_id)
print("----------------------------------")
print(df_users.iloc[our_user_id])
print("----------------------------------")
print(df_users.iloc[our_user_id]["essay"])
print("----------------------------------")
print(df_components.iloc[main_feature_id].nlargest(n=10))



# find best match for our user -- the top match should always be ourself
our_user = df_NMF.iloc[our_user_id]
similarities = df_NMF.dot(our_user)
match_user_id  = similarities.nlargest(n=1).index[0]
print("----------------------------------")
print("TOP MATCH SHOULD BE OURSELF:")
print("----------------------------------")
print(df_users.iloc[match_user_id])
print("----------------------------------")
print(df_users.iloc[match_user_id]["essay"])


# find 2nd best match for our user
match_user_id  = similarities.nlargest(n=2).index[1]
print("----------------------------------")
print("2nd MATCH:")
print("----------------------------------")
print(df_users.iloc[match_user_id])
print("----------------------------------")
print(df_users.iloc[match_user_id]["essay"])


# find 3rd best match for our user
match_user_id  = similarities.nlargest(n=3).index[2]
print("----------------------------------")
print("3rd MATCH:")
print("----------------------------------")
print(df_users.iloc[match_user_id])
print("----------------------------------")
print(df_users.iloc[match_user_id]["essay"])
print("------------------------------------------------")
print("Made recommendations in ",datetime.now() - startTime)
print("------------------------------------------------")
