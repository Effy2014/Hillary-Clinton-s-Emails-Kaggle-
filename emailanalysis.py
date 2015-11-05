
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from iso3166 import countries
import re
import networkx as nx
from gensim import corpora, models, similarities, utils
from nltk.corpus import stopwords
from nltk.corpus import names
from nltk import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
import string


# Get the general information of the dataset 

# In[3]:

con = sqlite3.connect('/Users/XW/Desktop/project/output/database.sqlite')
c = con.cursor()
c.execute("SELECT Name FROM sqlite_master WHERE type='table';")
tables = map(lambda t: t[0], c.fetchall())
for table in tables:
    rowsQuery = "SELECT Count() FROM %s" % table
    c.execute(rowsQuery)
    numberOfRows = c.fetchone()[0]
    print("%s\t%d" % (table, numberOfRows))


# In[3]:

Sender = pd.read_sql_query("""SELECT p.Name, count(p.Name) NumEmailsSent
                              FROM Emails e 
                              INNER JOIN Persons p ON e.SenderPersonId=P.Id
                              GROUP BY p.Name
                              ORDER BY COUNT(p.Name) DESC
                              LIMIT 10""", con)


# In[4]:

y_pos = np.arange(len(Sender.Name))[::-1]


# In[5]:

get_ipython().magic('matplotlib inline')
bar_width = 0.6
plt.barh(y_pos, Sender.NumEmailsSent, bar_width, color = "#53cfff")
plt.yticks(y_pos+bar_width/2, Sender.Name)
plt.xlabel('Number of Sent Emails')


# In[6]:

Receiver = pd.read_sql_query("""SELECT p.Name, COUNT(p.Name) NumEmailsReceived 
                                FROM Emails e 
                                INNER JOIN EmailReceivers r 
                                ON r.EmailId=e.Id 
                                INNER JOIN Persons p ON r.PersonId=p.Id 
                                GROUP BY p.Name 
                                ORDER BY COUNT(p.Name) DESC 
                                LIMIT 10""", con)


# In[7]:

y_pos = np.arange(len(Receiver.Name))[::-1]
get_ipython().magic('matplotlib inline')
bar_width = 0.6
plt.barh(y_pos, Receiver.NumEmailsReceived, bar_width, color = "#53cfff")
plt.yticks(y_pos+bar_width/2, Receiver.Name)
plt.xlabel('Number of Received Emails')


# In[8]:

Receiver1 = pd.read_sql_query("""SELECT p.Id, p.Name, COUNT(p.Name) NumEmailsReceived 
                                FROM Emails e 
                                INNER JOIN EmailReceivers r ON r.EmailId=e.Id 
                                INNER JOIN Persons p ON r.PersonId=p.Id 
                                GROUP BY p.Name 
                                ORDER BY COUNT(p.Name) DESC""", con)

Sender1 = pd.read_sql_query("""SELECT p.Id,p.Name, count(p.Name) NumEmailsSent
                              FROM Emails e 
                              INNER JOIN Persons p ON e.SenderPersonId=P.Id
                              GROUP BY p.Name
                              ORDER BY COUNT(p.Name) DESC""", con)


# In[9]:

tmp = pd.merge(Receiver1, Sender1, on='Id', how='outer')
temp = tmp.replace('NaN', 0)
temp['NumContact'] = (temp.NumEmailsReceived + temp.NumEmailsSent)


# In[10]:

Contact = temp.sort('NumContact', ascending = False)[:10][['Name_x', 'NumContact']]
Contact.columns = ['Name', 'NumContact']
Contact


# In[11]:

Contact.index = range(10)


# In[12]:

y_pos = np.arange(len(Contact.Name))[::-1]
get_ipython().magic('matplotlib inline')
bar_width = 0.6
plt.barh(y_pos, Contact.NumContact, bar_width, color = "#53cfff")
plt.yticks(y_pos+bar_width/2, Contact.Name)
plt.xlabel('Number of Total Contact Emails')


# In[13]:

emails = pd.read_sql_query("""SELECT ExtractedBodyText FROM Emails""", con)


# In[14]:

# remove empty emails 
emails = emails[emails["ExtractedBodyText"].str.len() > 0]
# convert emails to list for convenience
emails_body_text = emails["ExtractedBodyText"].tolist()


# In[15]:

# clean the emails body by remove the accents, dates, times, email address, web address 
def cleanEmailText(text):
    # Removes any accents
    text = utils.deaccent(text)
    # Replace hypens with spaces
    text = re.sub(r"-", " ", text)
    # Removes dates
    text = re.sub(r"\d+/\d+/\d+", "", text)
    # Removes times
    text = re.sub(r"[0-2]?[0-9]:[0-6][0-9]", "", text)
    # Removes email addresses
    text = re.sub(r"[\w]+@[\.\w]+", "", text)
    # Removes web addresses
    text = re.sub(r"/[a-zA-Z]*[:\/\/]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", "", text)
    # Remove any bad characters
    clndoc = ''
    for eachLetter in text:
        if eachLetter.isalpha() or eachLetter == ' ':
            clndoc += eachLetter
    text = ' '.join(clndoc.split())
    return text


# In[16]:

for i, item in enumerate(emails_body_text):
    emails_body_text[i] = cleanEmailText(item)
# get emails in a format that gensim can turn into a dictionary and corpus


# In[17]:

texts = [ [word for word in document.split() ] for document in emails_body_text]


# In[18]:

texts1 = texts.copy()
flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]
texts_count = flatten(texts1)
sw = set(stopwords.words('english'))
punctuation = set(string.punctuation)
all_names = set([name.lower() for name in names.words()])
def isStopWord(word):
    return (word in sw or word in punctuation) or not word.isalpha() or word in all_names
filtered = [w for w in texts_count if not isStopWord(w.lower())]
words = FreqDist(filtered)


# In[19]:

count = {}
for c in countries:
    country = c[0]
    count[country] = words[c[0]] + words[c[1]] + words[c[2]]


# In[20]:

flag = list(count.items())
country_occ = pd.DataFrame(flag, columns=['Country', 'Occurrence'])
MostMentioned = country_occ.sort('Occurrence', ascending = False)[:20]


# In[21]:

MostMentioned.index = range(20)
y_pos = np.arange(len(MostMentioned.Country))[::-1]
get_ipython().magic('matplotlib inline')
bar_width = 0.6
plt.barh(y_pos, MostMentioned.Occurrence, bar_width, color = "#53cfff")
plt.yticks(y_pos+bar_width/2, MostMentioned.Country)
plt.xlabel('Most Mentioned Countries of all Emails')


# In[24]:

filtered_lower = [w.lower() for w in texts_count if not isStopWord(w.lower())]
words = FreqDist(filtered_lower)


# In[25]:

texts2 = texts.copy()
flag = []
for fid in texts2:
    flag.append(" ".join([w.lower() for w in fid if not isStopWord(w.lower()) and words[w.lower()] > 1]))


# In[26]:

vectorizer = TfidfVectorizer(stop_words='english')
matrix = vectorizer.fit_transform(flag)
sums = np.array(matrix.sum(axis=0)).ravel()


# In[27]:

#Then let's say what topic is mentioned most in the emails
#remove the stop words and get the stem of the words 
ranks = []

for word, val in zip(vectorizer.get_feature_names(), sums):
    ranks.append((word, val))

df = pd.DataFrame(ranks, columns=["term", "tfidf"])
df = df.sort(['tfidf'], ascending = False)


# In[28]:

df[:50]


# In[ ]:



