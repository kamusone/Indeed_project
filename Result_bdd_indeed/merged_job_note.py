# -*- coding: utf-8 -*-


import numpy as np            # Data manipulation
import pandas as pd           # Dataframe manipulatio 
import matplotlib.pyplot as plt                   # For graphics

from sklearn import cluster, mixture              # For clustering
from sklearn.preprocessing import StandardScaler  # For scaling dataset
import nltk
%matplotlib inline
from nltk.corpus import stopwords
import string


df_job = pd.read_csv('merged_4jobs.csv')
df_job.Nom = df_job.Nom.str.upper()
df_note = pd.read_csv('merged_notes.csv')
df_note.Nom = df_note.Nom.str.upper()

df_merge =pd.merge(df_job, df_note, how='outer', left_on=['Titre','Nom'], right_on =['Titre','Nom'])
df_merge=df_merge.drop_duplicates()