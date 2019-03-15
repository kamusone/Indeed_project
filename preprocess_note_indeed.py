# -*- coding: utf-8 -*-
import numpy as np            # Data manipulation
import pandas as pd           # Dataframe manipulatio 
import matplotlib.pyplot as plt                   # For graphics

from sklearn import cluster, mixture              # For clustering
from sklearn.preprocessing import StandardScaler  # For scaling dataset
import nltk
%matplotlib inline

#load df
df_data_analyst = pd.read_csv('indeed_notes_infos_dataanalyst.csv')
df_data_scientist = pd.read_csv('indeed_notes_infos_datascientist.csv')
df_data_bi = pd.read_csv('indeed_notes_infos_bi.csv')
df_data_developpeur = pd.read_csv('indeed_notes_infos_developpeur.csv')

#drop duplicate

df_data_analyst=df_data_analyst.drop_duplicates()
df_data_scientist=df_data_scientist.drop_duplicates()
df_data_bi=df_data_bi.drop_duplicates()
df_data_developpeur=df_data_developpeur.drop_duplicates()

#drop NaN
df_data_analyst.dropna(how="all", inplace=True)
df_data_scientist.dropna(how="all", inplace=True)
df_data_bi.dropna(how="all", inplace=True)
df_data_developpeur.dropna(how="all", inplace=True)

#check isnull
df_data_analyst.isnull().sum()
df_data_scientist.isnull().sum()
df_data_bi.isnull().sum()
df_data_developpeur.isnull().sum()


#titre upper
df_data_analyst.Titre = df_data_analyst.Titre.str.upper()
df_data_analyst.Titre2 = df_data_analyst.Titre2.str.upper()

df_data_scientist.Titre = df_data_analyst.Titre.str.upper()
df_data_scientist.Titre2 = df_data_analyst.Titre2.str.upper()

df_data_bi.Titre = df_data_analyst.Titre.str.upper()
df_data_bi.Titre2 = df_data_analyst.Titre2.str.upper()

df_data_developpeur.Titre = df_data_analyst.Titre.str.upper()
df_data_developpeur.Titre2 = df_data_analyst.Titre2.str.upper()



#merge all jobs

df_all_note = pd.concat([df_data_analyst,df_data_scientist,df_data_bi,df_data_developpeur])
df_all_note.columns=["Titre","Titre2","Nom","Nom2","Avis","NoteGlobale","Équilibre_vie_professionnelle_personnelle","Salaire_Avantages_sociaux","Sécurité_emploi_Évolution_carrière","Management","Culture_d'entreprise"]
#create csv
df_all_note.to_csv("merged_notes.csv", encoding='utf-8', index=False)

## create a sub-df consisting only of jobs with annual salaries
df_all_note_avis=df_all_note[df_all_note.Avis.notnull()] # 1582 annual_salary
df_all_note_globale=df_all_note[df_all_note.NoteGlobale.notnull()] # 1582 annual_salary


df_mensuel_salary=df_all[df_all.Salaire.notnull()&df_all.Salaire.str.contains('par mois')]#203 mensuel_salary
