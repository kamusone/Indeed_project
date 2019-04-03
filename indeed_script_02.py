from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
import pandas as pd
import numpy as np
import re

##############################################################################################

def import_data_from_mongodb():
    try:
        client=MongoClient('mongodb://localhost:27017')
    except ConnectionFailure as e:
        print("Could not connect to MongoDB: %s" % e)
    db = client.Indeed_DataBase
    Indeed = db.Indeed
    Indeed = pd.DataFrame(list(Indeed.find()))
    return Indeed




##############################################################################################

langages=["PYTHON","C\s?\+\+","C\s?\#","VBA",
          "MATLAB","SPARK","GITHUB","AIRFLOW","POSTGRES","EC2","EMR","RDS",
          "REDSHIFT","AZURE","AWS","JAVA","HTML","CSS","JS","PHP","SQL",
          "RUBY","SAS","TCL","PERL","ORACLE","MYSQL","MONGODB","REDIS",
          "NEO4J","TWITTER","LDAP","FILE","HADOOP","DB2","SYBASE","AS400",
          "ACCESS","FLASK","BOOTSTRAP"]

##############################################################################################

diplomes = ['BTS', 'DUT', 'LICENCE', 'MASTER', 'DEUG', 'BAC\s?\+\s?1', 
            'BAC\s?\+\s?2', 'BAC\s?\+\s?3', 'BAC\s?\+\s?4', 'BAC\s?\+\s?5',
            'L\s?\3', 'M\s?1', 'M\s?2']

##############################################################################################

list_pattern = ["\s(\d)\sANS D’EXPERIENCE", "\s(\d)\sANS \(REQUIS\)", "\s(\d)\sAN \(REQUIS\)", 
                "\s(\d)\sANS \(SOUHAITE\)", "\s(\d)\sAN \(SOUHAITE\)", "\sEXPERIENCE DE\s(\d*)",
                "\sEXPERIENCE SIGNIFICATIVE DE\s(\d)", "\sEXPERIENCE PROFESSIONNELLE DE\s(\d)",
                "\sEXPERIENCE DE\s(\d)\sANNEE", "\sEXPERIENCE DE\s(\d)\sANNEES", "\sMOINS\s(\d)\sAN D'EXPERIENCE",
                "\sMOINS\s(\d)\sANNEES D'EXPERIENCE", "\sMOINS\s(\d)\sANNEE D'EXPERIENCE",
                "\s(\d)\sANNEES D'EXPERIENCE", "\s(\d)\sANNEE D'EXPERIENCE", "\s(\d)\sAN D’EXPERIENCE",
                "\s(\d)\sYEARS OF EXPERIENCE", "\s(\d)\sYEAR OF EXPERIENCE", "\sMOINS\s(\d)\sANS",
                "\sMOINS\s(\d)\sAN", "\/(\d)\sANS D’EXPERIENCE"]

##############################################################################################

list_debutant =["STAGE","STAGIAIRE","INTERNSHIP","TRAINEE", "ALTERNANCE"]

##############################################################################################

list_job = ['DATA SCIENTIST', 'DATA ANALYST', 'BUSINESS INTELLIGENCE', 'DEVELOPPEUR']

##############################################################################################

list_contrat= ['Apprentissage / Alternance', 'CDD', 'CDD, CDI', 'CDD, Freelance / Indépendant, CDI']

list_paris = ['PARIS', '75', '92', '93', '94', '95', '78', '77', '91',
              'ÎLEDEFRANCE', 'ILEDEFRANCE', 'HAUTSDESEINE', 'VALDEMARNE',
              'SEINESAINTDENIS']

##############################################################################################
def return_ville(adresse):
    for ville in list_paris:
        if ville in adresse:
            return 'Paris'
        else:
            pass
    
    if '69' in adresse or 'AUVERGNERHÔNEALPES' in adresse or 'RHÔNE' in adresse or '01' in adresse \
                       or 'SAINTQUENTINFALLAVIER' in adresse:
        return 'Lyon'
    else:
        pass
    if '31' in adresse or 'HAUTEGARONNE' in adresse or 'OCCITANIE' in adresse:
        return 'Toulouse'
    else:
        pass
    if '44' in adresse or 'LOIREATLANTIQUE' in adresse or 'PAYS DE LA LOIRE' in adresse:
        return 'Nantes'
    else:
        pass
    if '33' in adresse or 'GIRONDE' in adresse or 'FRANCE' in adresse or 'NOUVELLEAQUITAINE' in adresse:
        return 'Bordeaux'
    else:
        pass
    
##############################################################################################    

def add_ville_variable(df):
     for i in range(len(df.index)):
        adresse = str(df.loc[i,'Adresse'])
        ville = return_ville(adresse.upper())
        if ville:
            if ville== 'Paris':
                df.loc[i,'Ville'] = 5
            elif ville== 'Lyon':
                df.loc[i,'Ville'] = 4
            elif ville == 'Bordeaux' or ville == 'France':
                df.loc[i,'Ville'] = 3
            elif ville == 'Nantes':
                df.loc[i,'Ville'] = 2
            elif ville == 'Toulouse':
                df.loc[i,'Ville'] = 1
            else:
                df.loc[i,'Ville'] = 0       
     return df 


def add_ville(df):
     for i in range(len(df.index)):
        adresse = str(df.loc[i,'Adresse'])
        ville = return_ville(adresse.upper())
        if ville:
            if ville== 'Paris':
                df.loc[i,'Ville'] = 'Paris'
            elif ville== 'Lyon':
                df.loc[i,'Ville'] = 'Lyon'
            elif ville == 'Bordeaux' or ville == 'France':
                df.loc[i,'Ville'] = 'Bordeaux'
            elif ville == 'Nantes':
                df.loc[i,'Ville'] = 'Nantes'
            elif ville == 'Toulouse':
                df.loc[i,'Ville'] = 'Toulouse'
            else:
                df.loc[i,'Ville'] = np.nan      
     return df   
        
##############################################################################################
     
def add_lanages_variables(df):
    df['C'] = list(map(int, df["Descposte"].str.contains('[\s,.;](C)[\s*,.;!?]',regex=True, na=False)))
    df['R'] = list(map(int, df["Descposte"].str.contains('[\s,.;](R)[\s*,.;!?]',regex=True, na=False)))
    for langage in langages:
        df[langage] = list(map(int, df["Descposte"].str.contains(langage,regex=True, na=False)))
    
    df.rename(columns={ 'C\s?\+\+': 'langage C++',
                         'C\s?\#': 'langage C#'}, inplace=True)
                        
    return df

##############################################################################################

def add_job_variables(df):
     for job in list_job:
         df[job] = list(map(int, df["Titre"].str.contains(job,regex=True, na=False)))
                        
     return df
        
##############################################################################################           

def add_diplome_variables(df):
     for diplome in diplomes:
         df[diplome] = list(map(int, df["Descposte"].str.contains(diplome,regex=True, na=False)))
         df.rename(columns={'BAC\s?\+\s?1': 'BAC+1', 'BAC\s?\+\s?2': 'BAC+2', 
                             'BAC\s?\+\s?3': 'BAC+3', 'BAC\s?\+\s?4': 'BAC+4', 
                             'BAC\s?\+\s?5': 'BAC+5', 'L\s?\3': 'L3', 'M\s?1':'M1', 
                             'M\s?2': 'M2'}, inplace=True)
     return df 
      
##############################################################################################     
# Stagiaire débutant 0 année expérience
# Junior 1-2 années expériences
# Confirmé 3-4 années expériences
# Sénior 5-10 années expériences
# Expert >10 années expériences

def return_niveau_by_date(descreption):
    for pattern in list_pattern:
        code_niv = re.findall(pattern , descreption)
        if code_niv:
            code_niv = [x for x in code_niv if x]
            if (int(code_niv[0]) >= 1 and int(code_niv[0]) <= 2):
                return 'JUNIOR'
            elif (int(code_niv[0]) >= 3 and int(code_niv[0]) <= 4):
                return 'CONFIRME'
            elif (int(code_niv[0]) >= 5 and int(code_niv[0]) <= 10):
                return 'SENIOR'
            elif (int(code_niv[0]) > 10):
                return 'EXPERT'
            else:
                return 'DEBUTANT'
         
############################################################################################## 

def add_niveau_variables_by_date(df,i):
    descp = str(df.loc[i,'Descposte'])
    level = return_niveau_by_date(descp.upper())
    if level:
        if level== 'STAGIAIRE':
            df.loc[i,'level'] = 1
        elif level== 'DEBUTANT':
            df.loc[i,'level'] = 2
        elif level == 'JUNIOR':
            df.loc[i,'level'] = 3
        elif level == 'CONFIRME':
            df.loc[i,'level'] = 4
        elif level == 'SENIOR':
            df.loc[i,'level'] = 5
        elif level == 'EXPERT':
            df.loc[i,'level'] = 6
        else:
            df.loc[i,'level'] = 0
    else:
        df.loc[i,'level'] = 0
            
    return df
                
############################################################################################## 

def return_niveau_text(descreption, titre):
    if 'JUNIOR' in descreption or 'JUNIOR' in titre:
        return 'JUNIOR'
    elif 'SENIOR' in descreption or 'SENIOR' in titre:
        return 'SENIOR'
    elif 'DEBUTANT' in descreption or 'DEBUTANT' in titre:
        return 'DEBUTANT'
    else:
        for niveau in list_debutant:
            if niveau in descreption or niveau in titre:
                return "STAGIAIRE"
        else:
            pass 
 
############################################################################################## 
    
def add_niveau_variables_by_text(df):
    for i in range(len(df.index)):
        titre = str(df.loc[i,'Titre'])
        descp = str(df.loc[i,'Descposte'])
        level = return_niveau_text(descp.upper(), titre.upper())
        if level:
            if level== 'STAGIAIRE':
                df.loc[i,'level'] = 1
            if level== 'DEBUTANT':
                df.loc[i,'level'] = 2
            elif level == 'JUNIOR':
                df.loc[i,'level'] = 3
            elif level == 'CONFIRME':
                df.loc[i,'level'] = 4
            elif level == 'SENIOR':
                df.loc[i,'level'] = 5
            else:
                df.loc[i,'level'] = 0
        else:
            add_niveau_variables_by_date(df,i)
            
    return df

############################################################################################## 

def add_variables(df):
    X = add_niveau_variables_by_text(df)
    X = add_ville_variable(X)
    X = add_job_variables(X)
    X = add_diplome_variables(X)
    X = add_lanages_variables(X)

    return X

##############################################################################################
    
def return_features_model():
    df=pd.read_csv("indeedmongodb.csv")
    df = add_variables(df)
    df = df[~pd.isnull(df["Salaire_annuel_moyen"])]
    salaire_median = np.median(df["Salaire_annuel_moyen"])
    df["class_salaire"] = list(map((lambda x: "FAIBLE" if x < salaire_median else "HAUT"),
                                    df["Salaire_annuel_moyen"])) 
    df["label_class_salaire"] = df["class_salaire"].replace({"HAUT":1, "FAIBLE":0})
    return df 
 

##############################################################################################  

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import FrenchStemmer
fs =FrenchStemmer()
df = return_features_model()

df['ALTERNANCESTAGE'] = list(map(int, df["Contrat"].str.contains('ALTERNANCE|STAGE',regex=True, na=False)))
df['FREELANCE'] = list(map(int, df["Contrat"].str.contains('FREELANCE|INDÉPENDANT|INDEPENDANT',regex=True, na=False)))
df['CDI'] = list(map(int, df["Contrat"].str.contains('CDI',regex=True, na=False)))
df['CDD'] = list(map(int, df["Contrat"].str.contains('CDD',regex=True, na=False)))

df['Descposte']=[fs.stem(k) for k in df['Descposte']]


tfidf=TfidfVectorizer()
tfidf.fit_transform(df['Descposte'])

from sklearn.preprocessing import StandardScaler 

tfidf_col  = pd.DataFrame(tfidf.fit_transform(df['Descposte']).todense(),
             columns=tfidf.get_feature_names())



df= df.reset_index()
drop_columns = ['_id','Nom', 'Titre','Adresse', 'Salaire', 'Salaire_annuel_min', 
                'Salaire_annuel_max', 'Contrat', 'Descposte', 'Date', 
                'Salaire_annuel_moyen', 'class_salaire']

df_final = pd.merge(df.drop(columns=drop_columns, axis = 1),tfidf_col,right_index=True, left_index=True)
##############################################################################################
##Model RandomForest :

y = df_final["label_class_salaire"]
X = df_final.drop(["label_class_salaire"], axis=1)
   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model_rfc = RandomForestClassifier(n_estimators=300, random_state=90)
model_rfc.fit(X_train, y_train)
y_pred_rfc= model_rfc.predict(X_test)

accuracy_rfc =  accuracy_score(y_test, y_pred_rfc) * 100
print(accuracy_rfc)            
             
##############################################################################################
##Model SVM :

model_svm = SVC(probability=True,gamma='scale')
model_svm.fit(X_train, y_train)
y_pred_svm = model_svm.predict(X_test)

accuracy_svm =  accuracy_score(y_test, y_pred_svm) * 100
print(accuracy_svm)
        
##############################################################################################
#***************INDICATEURS DE PERFORMANCES****************************************************
#**********************************************************************************************
#Accuracy , pourcentage de bien classées
from sklearn.metrics import accuracy_score
accuracy_score_rfc=accuracy_score(y_test,y_pred_rfc)  #random forest
accuracy_score_svm=accuracy_score(y_test,y_pred_svm) #svm
print("pourcentage de bien classés pour le model rf", accuracy_score_rfc)
print("pourcentage de bien classés pour le model knn", accuracy_score_svm)

#+++++++++++++++++++++matrice de confusion 
from sklearn.metrics import confusion_matrix
confusion_matrix_rfc=confusion_matrix(y_test,y_pred_rfc)
confusion_matrix_svm=confusion_matrix(y_test,y_pred_svm)
print("matrice de confusion pour le model rf:",confusion_matrix_rfc, sep="\n")
print("matrice de confusion pour le model knn:", confusion_matrix_svm,sep="\n")

#++++++++++++++++++recall, precision e f1score
from sklearn.metrics import classification_report
print("rapport pour le model rf:",classification_report(y_test,y_pred_rfc), sep="\n")
print("rapport pour le model knn:",classification_report(y_test,y_pred_svm), sep="\n")

#++++++++++++++++++la courbe ROC with slearn and mtplotlib

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
#rappel: model_rfc=RandomForestClassifier()
#rappel: model_svm=SVM()

proba_rfc=model_rfc.predict_proba(X_test)[:,1]
proba_svm=model_svm.predict_proba(X_test)[:,1]

#cas model RFC
fpr, tpr, _ = roc_curve(y_test, proba_rfc)
plt.plot(fpr,tpr,"b-.", label="RFC")

#cas model svm
fpr, tpr, _ = roc_curve(y_test, proba_svm)
plt.plot(fpr,tpr,":", label="SVM")
#model aléatoire
plt.plot([[0,1],[0,1]],"r-", label="aléatoire")
#model parfait
plt.plot([[0,0,1],[0,0,1]],"b--", label="parfait")
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.legend()

############DATAVIZ***************************************************************************