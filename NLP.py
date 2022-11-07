import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import string
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import re

punc_str = string.punctuation+'\n'
#load data
train_df = pd.read_csv(r'C:\Users\zhuyu\PycharmProjects\nlp_tweets_disaster\train.csv')
sent_list = train_df['text'].str.lower().tolist()
#regrex pattern
http_reg = re.compile('http[s]?[^\s]*\s*')
uni_reg = re.compile(r'\\x[a-f0-9]{2}')
num_reg = re.compile('[0-9]')
multi_reg = re.compile(r'(.)\1{3,}')
# lowercase
for i in range(len(sent_list)):
    #removing punctuation
    tmp_str = sent_list[i].translate(str.maketrans('','',punc_str))
    # drop http
    tmp_str = http_reg.sub('',tmp_str)
    # drop unicode
    tmp_str = uni_reg.sub('',tmp_str)
    #drop numbers
    tmp_str = num_reg.sub('',tmp_str)
    #drop duplicates
    tmp2_str = multi_reg.sub(r'\1',tmp_str)

    tmp_list = sent_tokenize(tmp2_str)[0].split(' ')
    sent_list[i] = tmp_list
    #stemming
    '''stemmer = nltk.stem.porter.PorterStemmer()'''
    #lemmatization
#bag of words
#flatten word_list
#words = sum(sent_list,[])
#words = set(words)
#words_array = np.array(list(words))
mlb= MultiLabelBinarizer()
onehot = mlb.fit_transform(sent_list)

#bag of N Grams
count_vect = CountVectorizer(ngram_range=(2,2)) # range : sub string's length(amt of words)
sent_str_list = []
for sent in sent_list:
    sent_str_list.append(' '.join(sent))
boNg = count_vect.fit_transform(sent_str_list)
boNg_array = boNg.toarray()

#TF-IDF : Term Frequency
tfdif = TfidfVectorizer()
bow_tfdif = tfdif.fit_transform(sent_list)
