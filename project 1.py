#%%
pip install gensim
#%%
import re, jieba, pymysql
from gensim.models.word2vec import Word2Vec
import pandas as pd
db = pymysql.connect(host='rm-8vbwj6507z6465505ro.mysql.zhangbei.rds.aliyuncs.com',
                    port=3306,user='root',password='AI@2019@ai', database='stu_db')
cursor = db.cursor()
sql = 'select * from news_chinese'
cursor.execute(sql)
result = cursor.fetchall()
db.close()

reports = pd.Series([str(s).split(", ")[3] for s in result[1:]])

model = Word2Vec.load('word2vec.model')

word_similar = set()
model.wv.most_similar(['说'])
words = [s[0] for s in model.wv.most_similar(['说'])]
for w in words:
    word= [s[0] for s in model.wv.most_similar([w])]
    word_similar.update([w for w in word])

word_similar
#%%
remove = [ '！', '回家', '爱', '看来', '真的', '去', '你', '写', '什么', '看', '清楚', '得知', 
          '看法', '请', '回来', '呢', '我们', '深知', '发表', '介绍', '透露', '称该', ]
word_similar = [s for s in word_similar if s not in remove]


news_w_opinions = []
for article in reports:
    for w in word_similar:
        if w in article and '“' in article:
            news_w_opinions.append(article)
len(news_w_opinions)
#%%
len(reports)
#%%
news_w_opinions[0]

from pyltp import SentenceSplitter
import os
news_in_sentence = []
for news in news_w_opinions:
    news_in_sentence.append(SentenceSplitter.split(news))
news_in_words = []
for e in news_w_opinions:
    doc = re.sub('[０１２３４５６７８９a-zA-Z0-9\W]', '', e) # remove useless characters
    doc = list(jieba.cut(doc))
    news_in_words.append(doc)
list(news_in_sentence[0])

#%%
