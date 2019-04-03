# %%
import pandas as pd
import re
import numpy as np
import math
#path = 'indeed_bdd.csv'
data=pd.read_csv("indeedmongodb.csv")

#%%
#drop columns not used
dropping=['_id']
data.drop(dropping, axis=1, inplace= True)

#Str .UPPER Desctop & Contrat
data.Descposte=data.Descposte.str.upper()
data.Contrat=data.Contrat.str.upper()

#Calculate Minimum Salary by years/month/week/day/hour
def get_salaire_min_annuel(s):
    s = str(s)
    s = re.sub(r'(\d)\s+(\d)', r'\1\2', s)
    #s = s.replace("€","").replace("-"," ")
    s = s.replace("€","").replace("-"," ").replace(",",".")
    l_s = [float(sal) for sal in s.split() if sal.replace(".","").isdecimal()] 
    if "par an" in s:
        return l_s[0]
    elif "par mois" in s: 
        return l_s[0]*12
    elif "par semaine" in s: 
        return l_s[0]*52
    elif "par jour" in s: 
        return l_s[0]*(365-125)
    elif "par heure" in s: 
        return l_s[0]*8*(365-125)
    else:
        return np.nan


###############################################################################

##Calculate Maximum Salary by years/month/week/day/hour
    
def get_salaire_max_annuel(s):
    s = str(s)
    s = re.sub(r'(\d)\s+(\d)', r'\1\2', s)
    s = s.replace("€","").replace("-"," ").replace(",",".")
    l_s = [float(sal) for sal in s.split() if sal.replace(".","").isdecimal()] 
    if "par an" in s:
        s_a_max = l_s[1] if len(l_s)>1 else l_s[0] 
    elif "par mois" in s: 
        s_a_max = l_s[1]*12 if len(l_s)>1 else l_s[0]*12
    elif "par semaine" in s: 
        s_a_max = l_s[1]*52 if len(l_s)>1 else l_s[0]*52
    elif "par jour" in s: 
        s_a_max = l_s[1]*(365-125) if len(l_s)>1 else l_s[0]*(365-125)
    elif "par heure" in s: 
        s_a_max = l_s[1]*8*(365-125) if len(l_s)>1 else l_s[0]*8*(365-125)
    else:
        s_a_max = np.nan
    return s_a_max

###############################################################################
#Meam Salary   
def get_salaire_moyen_annuel(s):
    s = str(s)
    s = re.sub(r'(\d)\s+(\d)', r'\1\2', s)
    s = s.replace("€","").replace("-"," ").replace(",",".")
    l_s = [float(sal) for sal in s.split() if sal.replace(".","").isdecimal()] 
    salaire = np.mean(l_s)
    if "par an" in s:
        return salaire
    elif "par mois" in s: 
        return salaire*12
    elif "par semaine" in s: 
        return salaire*52
    elif "par jour" in s: 
        return salaire*(365-125)
    elif "par heure" in s: 
        return salaire*8*(365-125)
    else:
        return np.nan
    
#Clean Columns Contract Scraped all contracts

def contrat_clean_desc(contrat):
    try :
        if len(contrat) >70 or not re.findall(r'\bCDI\b|\bCDD\b|\bSTAGE\b|\bALTERNANCE\b|\bTEMPS PLEIN\b|\bTEMPS PARTIEL\b|\bFREELANCE\b|\bINDEPENDANT\b|\bINDÉPENDANT\b',contrat):
            return np.nan
        else:
            return contrat
    except :
        pass

#Extract Contract from description 
def contrat_clean(Descposte,contrat):
    try:
        if re.findall(r'\bCDI\b|\bCDD\b|\bSTAGE\b|\bALTERNANCE\b|\bTEMPS PLEIN\b|\bTEMPS PARTIEL\b|\bFREELANCE\b|\bINDEPENDANT\b|\bINDÉPENDANT\b',Descposte) and math.isnan(contrat):
            return  ' , '.join(re.findall(r'\bCDI\b|\bCDD\b|\bSTAGE\b|\bALTERNANCE\b|\bTEMPS PLEIN\b|\bTEMPS PARTIEL\b|\bFREELANCE\b|\bINDEPENDANT\b|\bINDÉPENDANT\b',Descposte))
        else :
            return contrat
    except:
        pass
    

#Clean contract contain CDI
def contrat_clean_CDI(contrat):
    try:
        if re.findall("CDI", contrat):
            return "CDI"
        else :
            return contrat
    except:
        pass
#Clean contract contain CDD
def contrat_clean_CDD(contrat):
    try:
        if re.findall("CDD", contrat):
            return "CDD"
        else :
            return contrat
    except:
        pass

#Extract kind diploma from Description 
def diplome(Descposte):
    try:
        if re.findall('BAC\s*\+\s*\d|BTS|DUT|LICENCE|MASTER|DEUG|M1|M2',Descposte):
            return  ' , '.join(re.findall('BAC\s*\+\s*\d|BTS|DUT|LICENCE|MASTER|DEUG|M1|M2',Descposte))
        else : 
            return np.nan
    except:
        pass
#Extract kind experience from Description 
def experience(Descposte):
    try :
        if re.findall(r'\bEXPERIENCE.*\d\s?ANS|\bEXPERIENCE.*\d\s?AN|\bEXPÉRIENCE.*\d\s?ANS|\bEXPÉRIENCE.*\d\s?AN|\bEXPERIENCE.*\d\s?YEARS|\bEXPERIENCE.*\d\s?YEAR|\bEXPERIENCE.*\d\s?Y',Descposte):
            return float(' - '.join(re.findall(r'\d',' , '.join(re.findall(r'\bEXPERIENCE.*\d\s?ANS|\bEXPERIENCE.*\d\s?AN|\bEXPÉRIENCE.*\d\s?ANS|\bEXPÉRIENCE.*\d\s?AN|\bEXPERIENCE.*\d\s?YEARS|\bEXPERIENCE.*\d\s?YEAR|\bEXPERIENCE.*\d\s?Y',Descposte))))[0])
        else :
            return np.nan
    except:
        pass
#Extract kind language from Description 
def langages(Descposte):
    try :
        if re.findall(r"PYTHON|[\s,.;]C[\s*,.;!?]|[\s,.;]R[\s*,.;!?]|C\+\+|C\s?#|VBA|MATLAB|SPARK|GITHUB|AIRFLOW|POSTGRES|EC2|EMR|RDS|REDSHIFT|AZURE|AWS|JAVA|HTML|CSS|JS|PHP|SQL|RUBY|SAS|TCL|PERL|ORACLE|MYSQL|MONGODB|REDIS|NEO4J|TWITTER|LDAP|FILE|HADOOP|DB2|SYBASE|AS400|ACCESS|FLASK|BOOTSTRAP",Descposte):
            return ' , '.join(set(' - '.join(re.findall(r"PYTHON|[\s,.;]C[\s*,.;!?]|[\s,.;]R[\s*,.;!?]|C\+\+|C\s?#|VBA|MATLAB|SPARK|GITHUB|AIRFLOW|POSTGRES|EC2|EMR|RDS|REDSHIFT|AZURE|AWS|JAVA|HTML|CSS|JS|PHP|SQL|RUBY|SAS|TCL|PERL|ORACLE|MYSQL|MONGODB|REDIS|NEO4J|TWITTER|LDAP|FILE|HADOOP|DB2|SYBASE|AS400|ACCESS|FLASK|BOOTSTRAP",Descposte)).split())).replace(' , -','').replace('- , ','')
        else :
            return np.nan
    except:
        pass
    
##Extract level experience from Description 

def niveau_desc(Descposte):
    try :
        if re.findall(r"JUNIOR|DÉBUTANT|DEBUTANT|SENIOR|EXPÉRIMENTÉ|EXPERIMENTE|EXPERIENCED",Descposte):
            s = ' - '.join(re.findall(r"JUNIOR|DÉBUTANT|DEBUTANT|SENIOR|EXPÉRIMENTÉ|EXPERIMENTE|EXPERIENCED",Descposte))
            return ' , '.join(set(s.split())).replace(' , -','').replace('- , ','')
        else :
            return np.nan
    except:
        pass

#Extract years experience from experience for complete information level experience

def niveau_annees_exp(experience):
    if experience>3:
        return 'SENIOR'
    elif  math.isnan(experience) :
        return  np.nan
    else:
        return'JUNIOR'
        
#complete info with match between level experience & years experience

def niveau(niveau_desc, niveau_exp):
    try:
        if re.findall("JUNIOR|DÉBUTANT|DEBUTANT", niveau_desc) and math.isnan(niveau_exp):
            return "JUNIOR"
        elif re.findall("SENIOR|EXPÉRIMENTÉ|EXPERIMENTE|EXPERIENCED", niveau_desc) and math.isnan(niveau_exp):
            return  "SENIOR"
    except:
        return niveau_exp


#Apply all fonction for any row of dataframe with create specific columns

data['salaire_min']=data.apply(lambda row : get_salaire_min_annuel(row['Salaire']), axis=1)
data['salaire_max']=data.apply(lambda row : get_salaire_max_annuel(row['Salaire']), axis=1)
data['salaire_mean']=data.apply(lambda row : get_salaire_moyen_annuel(row['Salaire']), axis=1)


data['Contrat']=data.apply(lambda row : contrat_clean_desc(row['Contrat']), axis=1)
data['Contrat']=data.apply(lambda row : contrat_clean(row['Descposte'],row['Contrat']), axis=1)
data['Contrat']=data.apply(lambda row : contrat_clean_CDI(row['Contrat']), axis=1)
data['Contrat']=data.apply(lambda row : contrat_clean_CDD(row['Contrat']), axis=1)

data['Diplome']=data.apply(lambda row : diplome(row['Descposte']), axis=1)


# Label-encoder for Diploma 
dip = ['BAC\s*\+\s*2','BAC\s*\+\s*3','BAC\s*\+\s*4','BAC\s*\+\s*5','BTS','DUT','LICENCE','MASTER','DEUG','M1','M2']

for v in dip :
    data[v] = list(map(int, data["Diplome"].str.contains(v,regex=True, na=False)))
    
data['Langages']=data.apply(lambda row : langages(row['Descposte']), axis=1)

# Label encoder for languages
lang = ["PYTHON","C\+\+","C\s?#","VBA","MATLAB","SPARK","GITHUB","AIRFLOW","POSTGRES","EC2","EMR","RDS","REDSHIFT","AZURE","AWS","JAVA","HTML","CSS","JS","PHP","SQL","RUBY","SAS","TCL","PERL","ORACLE","MYSQL","MONGODB","REDIS","NEO4J","TWITTER","LDAP","FILE","HADOOP","DB2","SYBASE","AS400","ACCESS","FLASK","BOOTSTRAP"]

for v in lang :
    data[v] = list(map(int, data["Langages"].str.contains(v,regex=True, na=False)))

# specific label encoder for C & R cause one letter.
data['C'] = list(map(int, data["Descposte"].str.contains('[\s,.;](C)[\s*,.;!?]',regex=True, na=False)))
data['R'] = list(map(int, data["Descposte"].str.contains('[\s,.;](R)[\s*,.;!?]',regex=True, na=False)))

#apply fonction for all rows
data['AnneesExperience']=data.apply(lambda row : experience(row['Descposte']), axis=1)
data['Niveau_annees_exp']=data.apply(lambda row : niveau_annees_exp(row['AnneesExperience']), axis=1)
data['Niveau_desc']=data.apply(lambda row : niveau_desc(row['Descposte']), axis=1)
data['Niveau']=data.apply(lambda row : niveau(row['Niveau_desc'],row['Niveau_annees_exp']), axis=1)



data['ALTERNANCE/STAGE'] = list(map(int, data["Contrat"].str.contains('ALTERNANCE|STAGE',regex=True, na=False)))
data['FREELANCE'] = list(map(int, data["Contrat"].str.contains('FREELANCE|INDÉPENDANT|INDEPENDANT',regex=True, na=False)))
data['CDI'] = list(map(int, data["Contrat"].str.contains('CDI',regex=True, na=False)))
data['CDD'] = list(map(int, data["Contrat"].str.contains('CDD',regex=True, na=False)))

#%%

#transform note on float
def repnote(note):
    try :
        note= note.replace(",",".")
        return float(note)
    except:
        pass

cols = ('NoteGlobale', 'Équilibre_vie_professionnelle_personnelle',
       'Salaire_Avantages_sociaux', 'Sécurité_emploi_Évolution_carrière',
       'Management', "Culture_d'entreprise")
for col in cols :
    data[col]=data.apply(lambda row : repnote(row[col]), axis=1)
#%%

# label-encoder for level // we suppose NaN is an information cause we suppose the entreprise give a chance for all 
def labelizer(niveau):
    niveau=str(niveau)
    if "JUNIOR" in niveau:
        return  float(1)
    elif "SENIOR" in niveau:
        return  float(2)
    else :
        return float(0)

data['Niveau_label']=data.apply(lambda row : labelizer(row['Niveau']), axis=1)

#drop duplicate

data = data.drop_duplicates()
data= data[data['salaire_max'].notnull()]
#%%
#create 
data['salaire_label'] = data['salaire_max']
data['salaire_label'][data['salaire_max'] < 48000] = 0
data['salaire_label'][data['salaire_max'] >= 48000] = 1

# data['salaire_label'] = data['salaire_max']
# data['salaire_label'][data['salaire_max'] < 30000] = 0
# data['salaire_label'][(data['salaire_max'] >= 30000) & (data['salaire_max'] <= 38000)] = 1
# data['salaire_label'][(data['salaire_max'] > 38000) & (data['salaire_max'] <= 40000)] = 2
# data['salaire_label'][(data['salaire_max'] > 40000) & (data['salaire_max'] <= 45000)] = 3
# data['salaire_label'][(data['salaire_max'] > 45000) & (data['salaire_max'] <= 50000)] = 4
# data['salaire_label'][(data['salaire_max'] > 50000) & (data['salaire_max'] <= 60000)] = 5
# data['salaire_label'][data['salaire_max'] > 60000] = 6

from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, StandardScaler, LabelEncoder
# data['salaire_label'] = LabelEncoder().fit_transform(data[['salaire_label']])
data['salaire_label'] = LabelBinarizer().fit_transform(data[['salaire_label']])
data['Niveau_label'] = LabelEncoder().fit_transform(data[['Niveau_label']])

#%%

cols = ['NoteGlobale', 'Équilibre_vie_professionnelle_personnelle',
       'Salaire_Avantages_sociaux', 'Sécurité_emploi_Évolution_carrière',
       'Management', "Culture_d'entreprise"]
for c in cols:
    data[c].fillna(-999,inplace=True)
data['Avis'].fillna(0,inplace=True)

#%%
data.dropna(subset=["Descposte"], axis=0, inplace=True)




#%%
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import FrenchStemmer
fs =FrenchStemmer()
data['Descposte']=[fs.stem(k) for k in data['Descposte']]


tfidf=TfidfVectorizer()
tfidf.fit_transform(data['Descposte'])

from sklearn.preprocessing import StandardScaler 

cvec_table  = pd.DataFrame(tfidf.fit_transform(data['Descposte']).todense(),
             columns=tfidf.get_feature_names())


#%%
data= data.reset_index()


include = data.columns.drop(['Titre', 'Nom', 'Adresse', 'Salaire', 'Contrat', 'Descposte', 'Date', 'salaire_min','salaire_mean', 'salaire_max', 'Diplome',  'Langages' ,'AnneesExperience','Niveau_annees_exp', 'Niveau_desc', 'Niveau', 'NoteGlobale', 'Équilibre_vie_professionnelle_personnelle'
                             , 'Sécurité_emploi_Évolution_carrière', 'Management', "Culture_d'entreprise"])
new_df = pd.merge(data[include],cvec_table,right_index=True, left_index=True)

from sklearn.model_selection import cross_val_score, train_test_split

X = new_df
y = data.salaire_label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



#%% SVM

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
model = SVC()
model.fit(X_train, y_train)
y_pred_svc = model.predict(X_test)

accuracy_svc =  accuracy_score(y_test, y_pred_svc) * 100
print(accuracy_svc)

#%% Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred_rfc = model.predict(X_test)

accuracy_rfc =  accuracy_score(y_test, y_pred_rfc) * 100
print(accuracy_rfc)