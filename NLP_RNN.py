#!/usr/bin/env python
# coding: utf-8

# In[13]:


import sys
import pandas as pd
import numpy as np


# In[10]:


#COVID-CT Reports
COVIDxl= pd.ExcelFile('/Users/yangs/Desktop/Shuang/phd courses/2021spring/BME6938_multimodal/final project/COVID-CT-master/COVID-CT-MetaInfo.xlsx')
COVIDxl.sheet_names

for sheet in COVIDxl.sheet_names:
    file= pd.read_excel(COVIDxl,sheet_name= sheet, usecols="H, K")
    file.to_csv(sheet + '.txt', header= True, index= False)


# In[7]:


import collections
import matplotlib.pyplot as plt
# Read input file, note the encoding is specified here 
# It may be different in your text file
file = open('positive_captions.txt', encoding="utf8")
a= file.read()
# Stopwords
stopwords = set(line.strip() for line in open('stopwords.txt', encoding="utf8"))
stopwords = stopwords.union(set(['Figure', 'Patient', 'China', 'Wuhan', 'Beijing']))
# Instantiate a dictionary, and for every word in the file, 
# Add to the dictionary if it doesn't exist. If it does, increase the count.
wordcount = {}
# To eliminate duplicates, remember to split by punctuation, and use case demiliters.
for word in a.lower().split():
    word = word.replace(".","")
    word = word.replace(",","")
    word = word.replace(":","")
    word = word.replace("\"","")
    word = word.replace("!","")
    word = word.replace("â€œ","")
    word = word.replace("â€˜","")
    word = word.replace("*","")
    if word not in stopwords:
        if word not in wordcount:
            wordcount[word] = 1
        else:
            wordcount[word] += 1
# Print most common word
n_print = int(input("How many most common words to print: "))
print("\nOK. The {} most common words are as follows\n".format(n_print))
word_counter = collections.Counter(wordcount)
for word, count in word_counter.most_common(n_print):
    print(word, ": ", count)
# Close the file
file.close()


# In[11]:


# Create a data frame of the most common words 
# Draw a bar chart
lst = word_counter.most_common(n_print)
df_rank = pd.DataFrame(lst, columns = ['Word', 'Count'])
df_rank.plot.bar(x='Word',y='Count')


# In[14]:


number_of_words = df_rank.shape[0]
#number_of_words 611
df_rank['word_index'] = list(np.arange(number_of_words)+1)


# In[15]:


df_rank


# In[16]:


word_dict=df_rank.set_index('Word')['word_index'].to_dict()


# In[20]:


df = pd.read_excel (r'/Users/yangs/Desktop/Shuang/phd courses/2021spring/BME6938_multimodal/final project/COVID-CT-master/COVID-CT-MetaInfo.xlsx')


# In[23]:


df['Severity_Group'] = np.where(df['Severity'].str.contains("Severe"or "mortality" or "Serious" or "critical" or "severe" or "Mortality" or "Critical" or "serious"), '1', '0')


# In[24]:


df.head(2)


# In[25]:


df['Severity_Group'].value_counts()


# In[26]:


import nltk
from nltk.tokenize import word_tokenize


# In[27]:


df['Captions'] = df['Captions'].astype('str') 
df['Captions_token']= df['Captions'].apply(word_tokenize)


# In[31]:


def word_index(df, col='Captions',dictionary=word_dict):
    word_index=[]
    for item in df[col]:
        wtoken=word_tokenize(item)
        mp = list(map(dictionary.get, wtoken))
        my_list = [ 0 if i is None else i for i in mp]
        #my_array = np.array(my_list)
        #print(my_array)
        word_index.append(my_list )
        #word_index=pd.concat([word_index, my_array])
    return word_index


# In[32]:


token_lst= word_index(df, col='Captions',dictionary=word_dict)


# ## Data Preparation

# In[34]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers


# In[36]:


data_array = np.array(token_lst)


# In[37]:


np.unique(df['Severity_Group'])


# In[39]:


len(np.unique(np.hstack(data_array)))


# In[40]:


length = [len(i) for i in data_array]
print("Average Review length:", np.mean(length))
print("Standard Deviation:", round(np.std(length)))


# In[74]:


def vectorize(sequences, dimension = 650):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


# In[75]:


data_vec=vectorize(data_array, dimension = 650)
targets = np.array(df['Severity_Group']).astype("float32")


# In[76]:


test_x = data_vec[:105]
test_y = targets[:105]
train_x = data_vec[105:]
train_y = targets[105:]


# In[77]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[91]:


model = keras.Sequential()


# In[92]:


# Input - Layer
model.add(layers.Dense(20, activation = "relu", input_shape=(650, )))
# Hidden - Layers
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(20, activation = "relu"))

model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(20, activation = "relu"))
# Output- Layer
model.add(layers.Dense(1, activation = "sigmoid"))
model.summary()


# In[93]:


model.compile(
 optimizer = "adam",
 loss = "binary_crossentropy",
 metrics = ["accuracy"]
)


# In[94]:


results = model.fit(
 train_x, train_y,
 epochs= 2,
 batch_size = 50,
 validation_data = (test_x, test_y)
)


# In[95]:


print("Test-Accuracy:", np.mean(results.history["val_accuracy"]))


# In[ ]:




