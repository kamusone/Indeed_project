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

#import & rename columns
df_analyst = pd.read_csv('merged_data_analyst.csv')
df_analyst.columns=["Titre","Nom","Adresse","Salaire","Contrat","Descpost","Date","Metier"]
df_bi = pd.read_csv('merged_BI.csv')
df_bi.columns=["Titre","Nom","Adresse","Salaire","Contrat","Descpost","Date","Metier"]
df_scientist = pd.read_csv('merged_data_scientist.csv')
df_scientist.columns=["Titre","Nom","Adresse","Salaire","Contrat","Descpost","Date","Metier"]
df_developeur = pd.read_csv('merged_developeur.csv')
df_developeur.columns=["Titre","Nom","Adresse","Salaire","Contrat","Descpost","Date","Metier"]



#merged all csv
df_all = pd.concat([df_analyst,df_bi,df_scientist,df_developeur])

#create csv
df_all.to_csv("merged_4jobs.csv", encoding='utf-8', index=False)


## create a sub-df consisting only of jobs with annual salaries
df_annual_salary=df_all[df_all.Salaire.notnull()&df_all.Salaire.str.contains('par an')] # 1582 annual_salary

df_mensuel_salary=df_all[df_all.Salaire.notnull()&df_all.Salaire.str.contains('par mois')]#203 mensuel_salary
df_all.Salaire = df_all.Salaire.astype(str)