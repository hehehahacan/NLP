#%%
import pandas as pd
import jieba
token=[]
with open('wiki_02simple','r',encoding='utf-8') as f:
    line=f.readlines()
    for i, line in enumerate(line):
        if i % 10000   == 0:
            print (i)
        if i%500000  ==0 and i!=0:
            break
        words =list(jieba.cut(line))
        token.append(words)
#%%
token[:100]

#%%
from gensim.models import word2vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
%matplotlib inline

#%%
model = word2vec.Word2Vec(token, size=100, window=20, min_count=200, workers=8)


#%%
model.wv.similarity('百姓', '参加')

#%%
model.wv['中学']  

#%%
def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

#%%
matplotlib.rcParams['font.family'] = 'Arial Unicode MS'
matplotlib.rcParams['font.serif'] = 'Arial Unicode MS'
matplotlib.rcParams['font.sans-serif'] = 'Arial Unicode MS'
matplotlib.rcParams['axes.unicode_minus'] = False

#%%
tsne_plot(model)