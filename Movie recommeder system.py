#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[4]:


import numpy as np


# In[5]:


movies=pd.read_csv("tmdb_5000_movies.csv")
credits=pd.read_csv("tmdb_5000_credits.csv")


# In[6]:


movies.head(1)


# In[7]:


credits.head(1)


# In[8]:


#Merging data


# In[9]:


#1


# In[10]:


movies=pd.merge(movies,credits, on="title")


# In[11]:


movies.shape


# In[12]:


#2


# In[13]:


movies.merge(credits,on="title")


# In[14]:


df=movies[['genres','movie_id','cast','crew','overview','title','keywords']]


# In[15]:


df


# In[16]:


df.isnull().sum()


# In[17]:


df.dropna(inplace=True)


# In[18]:


df.duplicated().sum()


# In[19]:


df.iloc[0].genres


# In[20]:


#these are in string of list  ,convert them into list


# In[21]:


#This is called helper function


# In[26]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
       L.append(i['name'])
    return L


# In[27]:


'''
The convert function is designed to take a string obj
that represents a list of dictionaries (presumably with a key 'name')
and convert it into a new list containing only the values associated with 
the 'name' key.
'''


# In[31]:


convert('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
)


# In[30]:


import ast 
ast.literal_eval


# In[32]:


df['genres'].apply(convert)


# In[33]:


df['genres']=df['genres'].apply(convert)


# In[34]:


df


# In[35]:


df.iloc[0].keywords


# In[36]:


df['keywords']=df['keywords'].apply(convert)


# In[80]:


def convert3(obj):
    L=[]
    counter = 0
    for i in ast.literal_eval(obj):
        if counter !=3:
         L.append(i['name'])
         counter+=1
    else:
        break
    return L


# In[37]:


df.iloc[0].cast


# In[38]:


import ast

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L


# In[39]:


df['cast']=df['cast'].apply(convert3)


# In[84]:


df


# In[86]:


df.iloc[0].crew


# In[92]:


def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
       if i['job'] == 'Director': 
         L.append(i['name'])
         break
    return L


# In[94]:


df['crew']=df['crew'].apply(fetch_director)


# In[95]:


df


# In[100]:


df['overview']=df['overview'].apply(lambda x: x.split())


# In[101]:


df


# In[103]:


columns_to_process = ['genres', 'keywords', 'cast', 'crew']

for column in columns_to_process:
    df[column] = df[column].apply(lambda x: [name.replace(' ', '') for name in x])


# In[104]:


df


# In[107]:


df['hashtags'] = df.apply(lambda row: ' '.join(map(str, row[['cast', 'crew', 'overview', 'genres', 'keywords']])), axis=1)


# In[109]:


df


# In[111]:


df.iloc[0]['hashtags']


# In[113]:


df


# In[128]:


new_df=df[['movie_id','title','hashtags']]


# In[129]:


new_df['hashtags'] = new_df['hashtags'].apply(lambda x: x.lower())


# In[130]:


new_df


# In[131]:


new_df['hashtags'] = new_df['hashtags'].apply(lambda x: x.replace("'", ""))


# In[132]:


new_df


# In[142]:


new_df['hashtags'][1]


# In[134]:


import re


# In[135]:


new_df['hashtags'] = new_df['hashtags'].apply(lambda x: re.sub(r"[\[\],']", "", x))


# In[136]:


new_df['hashtags'][1]


# In[177]:


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words='english')


# In[178]:


vectors=cv.fit_transform(new_df['hashtags']).toarray()


# In[179]:


cv.get_feature_names_out()


# In[180]:


cv.get_feature_names_out().tolist()


# In[158]:


get_ipython().system('pip install nltk')


# In[159]:


import nltk


# In[172]:


from nltk.stem.porter  import PorterStemmer
ps = PorterStemmer()


# In[173]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
        
    return ' '.join(y)


# In[175]:


new_df['hashtags']=new_df['hashtags'].apply(stem)


# In[181]:


cv.get_feature_names_out().tolist()


# In[182]:


from sklearn.metrics.pairwise  import cosine_similarity


# In[187]:


similarity=cosine_similarity(vectors)


# In[185]:


cosine_similarity(vectors).shape


# In[188]:


similarity[0]


# In[189]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])


# In[194]:


def recommend (movie):
    movie_index=new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list =sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
        
    for i in movie_list: 
           print(new_df.iloc[i[0]].title)
        


# In[195]:


recommend("Batman Begins")


# In[ ]:




