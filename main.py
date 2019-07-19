# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 12:28:14 2019

@author: Pranay Rawat
"""

import requests
import pandas as pd
import json

pd.set_option('display.max_colwidth' , 200)

auth_params = {'consumer_key':'86550-50025b552add00bc37c2afa8','redirect_uri':'https://github.com/Pranay10rawat'}
tkn = requests.post('https://getpocket.com/v3/oauth/request' ,data=auth_params)
tkn.content


usr_params = {'consumer_key':'86550-50025b552add00bc37c2afa8','code':'9ab91df8-ae9a-2d65-4fbc-2dd037'}
usr = requests.post('https://getpocket.com/v3/oauth/authorize' ,data=usr_params)
usr.content



#extracting unliked stories body 
no_params={'consumer_key':'86550-50025b552add00bc37c2afa8','access_token':'3ac5ad89-7d22-7de0-85c7-b580e9','tag':'unliked'}
no_result = requests.post('https://getpocket.com/v3/get',data=no_params)
no_result.text



#extracting urls from the extracted bodies
#for stories tages as unliked
no_jf = json.loads(no_result.text)
no_jd = no_jf['list']
no_urls = []

for i in no_jd.values():
    no_urls.append(i.get('resolved_url'))
no_urls

no_uf = pd.DataFrame(no_urls , columns=['urls'])
no_uf = no_uf.assign(wanted = lambda x:'unliked')
no_uf

#liked stories
yes_params={'consumer_key':'86550-50025b552add00bc37c2afa8','access_token':'3ac5ad89-7d22-7de0-85c7-b580e9','tag':'liked'}
yes_result = requests.post('https://getpocket.com/v3/get',data=yes_params)
yes_result.text


#extracting urls of the liked stories
yes_jf = json.loads(yes_result.text)
yes_jd = yes_jf['list']
yes_urls = []



for i in yes_jd.values():
    yes_urls.append(i.get('resolved_url'))
yes_urls

yes_uf = pd.DataFrame(yes_urls , columns=['urls'])
yes_uf = yes_uf.assign(wanted = lambda x:'liked')
yes_uf

#dropna deletes rows and columns with null values
df = pd.concat([yes_uf,no_uf])
df.dropna(inplace=True)
df


#Using the embed.ly API to download story bodies
import urllib
def get_html(x):
    qurl = urllib.parse.quote(x)
    rhtml = requests.get('https://api.embedly.com/1/extract?url=' + qurl + '&key=1e6df8c5ba3f4f7eb761c987f5a9f69a')
    ctnt = json.loads(rhtml.text).get('content')
    return ctnt

df.loc[:,'html'] = df['urls'].map(get_html)
df.dropna(inplace=True)
df

from bs4 import BeautifulSoup
def get_text(x):
    soup = BeautifulSoup(x,'lxml')
    text=soup.get_text()
    return text

df.loc[:,'text']= df['html'].map(get_text)
df

#Natural language processing basics

from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(ngram_range = (1,3),stop_words = 'english',min_df=2)
tv=vect.fit_transform(df['text'])
tv


#support vector machines
from sklearn.svm import LinearSVC
clf = LinearSVC()
model = clf.fit(tv,df['wanted'])

import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials


scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name(r'C:\Users\Pranay Rawat\Downloads\client.json', scope)
gc = gspread.authorize(credentials)


ws = gc.open("DailyNews")
sh = ws.sheet1
zd = list(zip(sh.col_values(2),sh.col_values(3),sh.col_values(4)))
zf = pd.DataFrame(zd , columns =['title','urls','html'])
zf.replace('',pd.np.nan,inplace=True)
zf.dropna(inplace = True)
zf

zf.loc[:,'text'] = zf['html'].map(get_text)
zf.reset_index(drop=True,inplace=True)
test_matrix = vect.transform(zf['text'])
test_matrix

results = pd.DataFrame(model.predict(test_matrix),
columns = ['wanted'])
results

rez = pd.merge(results,zf,left_index=True , right_index = True)
rez

combined = pd.concat([df[['wanted','text']] , rez[['wanted','text']]])
combined

tvcomb = vect.fit_transform(combined['text'],combined['wanted'])
model = clf.fit(tvcomb,combined['wanted'])

import pickle 
pickle.dump(model,open(r'M:\MachineLearningProjects\my\CustomNewsFeed\news_model_pickle.p','wb'))
pickle.dump(vect , open(r'M:\MachineLearningProjects\my\CustomNewsFeed\news_vect_pickle.p','wb'));


