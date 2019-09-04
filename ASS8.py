#%%
import jieba
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
warnings.filterwarnings('ignore')
%matplotlib notebook

#%%
content = pd.read_csv('sqlResult_1558435.csv', encoding='gb18030')

#%%
content.head()

#%%
content = content.dropna(subset=['source', 'content'])
content['xinhua'] = content.source.apply(lambda x: 1 if '新华社' in x else 0) 
content.head()
#%%
X,y = content[['content']],content[['xinhua']]
#%%
new_X = []
for i in range(len(X)):
    tmp = re.sub('[\\a-zA-Z0-9，。（）/：…@！？\s\n]', '', str(X.iloc[i, 0]))
    new_X.append(' '.join(jieba.cut(tmp)))
new_X = pd.Series(new_X)
#%%
new_X[10]

#%%
 from sklearn.feature_extraction.text import TfidfVectorizer
 vectorizer = TfidfVectorizer()
 X = vectorizer.fit_transform(new_X)


#%%
X.shape

#%%

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
x_train, x_valid, y_train, y_valid = train_test_split(X, y, random_state=45, test_size=0.15)

#%%
#SVM
svc = SVC(verbose=5)
svc.fit(x_train, y_train)

#%%
y_pred = svc.predict(x_valid)
y_pred_proba = svc.decision_function(x_valid)

#%%
svc.score(x_valid, y_valid)

#%%
#Random Forest
rfc = RandomForestClassifier(oob_score=True, class_weight='balanced', verbose=5, random_state=42, n_jobs=4)
rfc.fit(x_train, y_train)

#%%
y_pred = rfc.predict(x_valid)   
y_pred_proba = rfc.predict_proba(x_valid)

#%%
rfc.score(x_valid, y_valid)

#%%
precision_score(y_valid, y_pred)

#%%
rfc = RandomForestClassifier(n_estimators=15, oob_score=True, class_weight='balanced', verbose=5, random_state=42, n_jobs=4)
rfc.fit(x_train, y_train)

#%%
rfc.score(x_valid, y_valid)

#%%
precision_score(y_valid, y_pred)

#%%
