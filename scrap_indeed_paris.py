# -*- coding: utf-8

import numpy as np            # Data manipulation
import pandas as pd           # Dataframe manipulatio 
import matplotlib.pyplot as plt                   # For graphics

from sklearn import cluster, mixture              # For clustering
from sklearn.preprocessing import StandardScaler  # For scaling dataset

#import plotly.graph_objs as go
#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplotinit_notebook_mode(connected=True) #additional initialization step to plot offline in Jupyter Notebooks


def explore_dataset(data):
    ds = data
    print("\nData set Attributes:\n")
    print("\nShape:\n",ds.shape)
    print("\nColumns:\n",ds.columns.values)
    print("\n1st 2 rows:\n",ds.head(2))
    print("\nData type:\n",ds.dtypes)
    #print("\nDataset info:\n",ds.info)
    print("\nDataset summary:\n",ds.describe())  
#%%  PARIS
    
df1 = pd.read_csv('scrappingindeed_paris.csv')
df2 = pd.read_csv('scrappingindeed_paris_2.csv')
df3 = pd.read_csv('scrappingindeed_paris_3_p55.csv')
df4 = pd.read_csv('scrappingindeed_paris_4_p100.csv')
#
#explore_dataset(df1)

df1_drop=df1.drop_duplicates()
(df1["Salaire"].count()/len(df1.index))/100
df2_drop=df2.drop_duplicates()
(df2["Salaire"].count()/len(df2.index))/100
df3_drop=df3.drop_duplicates()
(df3["Salaire"].count()/len(df3.index))/100
df4_drop=df4.drop_duplicates()
(df4["Salaire"].count()/len(df4.index))/100


#merge df paris
merged_df_paris = pd.concat([df1_drop, df2_drop,df3_drop,df4_drop])
#save on csv
merged_df_paris.to_csv("merged_df_paris.csv", encoding='utf-8', index=False)

test_paris=pd.read_csv("merged_df_paris.csv")
test_paris.head()

#%%  LYON
df1 = pd.read_csv('scrappingindeed_lyon_12.csv')
df2 = pd.read_csv('scrappingindeed_lyon_15.csv')
df3 = pd.read_csv('scrappingindeed_lyon_23.csv')
df4 = pd.read_csv('scrappingindeed_lyon_24.csv')

df1_drop=df1.drop_duplicates()
(df1["Salaire"].count()/len(df1.index))/100
df2_drop=df2.drop_duplicates()
(df2["Salaire"].count()/len(df2.index))/100
df3_drop=df3.drop_duplicates()
(df3["Salaire"].count()/len(df3.index))/100
df4_drop=df4.drop_duplicates()
(df4["Salaire"].count()/len(df4.index))/100
#df1_dup.dropna(axis=0, how='all', inplace=True)
#merge df paris
merged_df_lyon = pd.concat([df1_drop, df2_drop,df3_drop,df4_drop])
#save on csv
merged_df_lyon.to_csv("merged_df_lyon.csv", encoding='utf-8', index=False)

test_lyon=pd.read_csv("merged_df_lyon.csv")
test_lyon.head()
#%%      Bordeaux
df1 = pd.read_csv('scrappingindeed_bordeaux_7.csv')


df1_drop=df1.drop_duplicates()
(df1["Salaire"].count()/len(df1.index))/100
merged_df_bordeaux=df1_drop
#save on csv
merged_df_bordeaux.to_csv("merged_df_bordeaux.csv", encoding='utf-8', index=False)

test_bordeaux=pd.read_csv("merged_df_bordeaux.csv")
test_bordeaux.head()
#%% Nantes
df1 = pd.read_csv('scrappingindeed_nantes_9.csv')


df1_drop=df1.drop_duplicates()
(df1["Salaire"].count()/len(df1.index))/100
merged_df_nantes=df1_drop
#%%
#save on csv
merged_df_nantes.to_csv("merged_df_nantes.csv", encoding='utf-8', index=False)

test_nantes=pd.read_csv("merged_df_nantes.csv")
test_nantes.head()
#%% toulouse
df1 = pd.read_csv('scrappingindeed_toulouse_16.csv')


df1_drop=df1.drop_duplicates()
merged_df_toulouse=df1_drop

merged_df_toulouse.to_csv("merged_df_toulouse.csv", encoding='utf-8', index=False)

test_toulouse=pd.read_csv("merged_df_toulouse.csv")
test_toulouse.head()

#%%
#Merge data 

df1 = pd.read_csv('merged_df_paris.csv')
df2 = pd.read_csv('merged_df_lyon.csv')
df3 = pd.read_csv('merged_df_bordeaux.csv')
df4 = pd.read_csv('merged_df_nantes.csv')
df5 = pd.read_csv('merged_df_toulouse.csv')

merged_BI=pd.concat([df1, df2,df3,df4,df5])
merged_BI.dropna(axis=0, how='all',inplace=True)
#save to csv
merged_BI.to_csv("merged_BI.csv", encoding='utf-8', index=False)




