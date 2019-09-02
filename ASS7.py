#%%
import pandas as pd
content = pd.read_csv('sqlResult_1558435.csv', encoding='gb18030')
#%%
content = content.dropna(subset=['source', 'content'])
len(content)
#%%
content.head()
#%%
content['xinhua'] = content.source.apply(lambda x: 1 if '新华社' in x else 0) 
#%%
content.head()
#%%
X,y = content[['content']],content[['xinhua']]
#%%
import jieba
import re
X[:10]
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
print(X.shape)

#%%
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
#%%
#LogisticRegression
X_train, x_valid, y_train, y_valid = train_test_split(X,y,random_state=1002, test_size=0.15)
#%%
lr = LogisticRegression() 
reg=lr.fit(X_train, y_train)

#%%
reg.coef_

#%%
reg.intercept_

#%%
lr.score(x_valid, y_valid)

#%%
#Naive Bayes
x_train, x_valid, y_train, y_valid = train_test_split(X, y, random_state=62, test_size=0.15)
#%%
knc = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
knc.fit(x_train, y_train)

#%%
knc.score(x_valid, y_valid)

#%%
precision_score(y_valid, y_pred)
#%%
y_pred = lr.predict(X)
y_pred.shape, y.shape

#%%
len(content)
#%%
content['y_pred'] = y_pred
content.head(10)

#%%
copy_news = content[(content.xinhua == 0) & (content.y_pred == 1)]
copy_news.head(2)

#%%
len(copy_news.source)

#%%
copy_sources = set(copy_news.source.to_list())

#%%
len(copy_sources)

#%%
len(copy_news)

#%%
copy_news.head(20)

#%%
