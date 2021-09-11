#!/usr/bin/env python
# coding: utf-8

# # K nearest Neighbors

# In[45]:


import numpy as np 
import pandas as pd
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split


# In[82]:


df =pd.read_csv("C:/Users/A Emiliano Fragoso/Desktop/MLcourse/python-ml-course-master/datasets/cancer/breast-cancer-wisconsin.data.txt", header=None)


# In[83]:


df.head()


# In[ ]:





# In[84]:


df.columns=["name","V1","V2","V3","V4","V5","V6","V7","V8","V9","class"]


# In[85]:


df.head()


# In[86]:


df=df.drop(["name"],1)


# In[ ]:





# In[87]:


df.replace("?", -99999, inplace=True)


# In[ ]:





# In[88]:


Y=df["class"]


# In[ ]:





# In[89]:


X=df[["V1","V2","V3","V4","V5","V6","V7","V8","V9"]]


# In[ ]:





# In[90]:


X.head()


# In[ ]:





# In[91]:


Y.head()


# In[ ]:





# # Clasificador de los K vecinos

# In[92]:


Xtrain, Xtest, Ytrain, Ytest=train_test_split(X,Y, test_size=0.2)


# In[ ]:





# In[93]:


clf = neighbors.KNeighborsClassifier()


# In[ ]:





# In[94]:


clf.fit(Xtrain, Ytrain)


# In[ ]:





# In[95]:


accuracy=clf.score(Xtest, Ytest)
accuracy


# # Clasificacion sin limpieza

# In[60]:


df=pd.read_csv("C:/Users/A Emiliano Fragoso/Desktop/MLcourse/python-ml-course-master/datasets/cancer/breast-cancer-wisconsin.data.txt", header=None)


# In[61]:


df.replace("?", -99999, inplace=True)
df.columns=["name","V1","V2","V3","V4","V5","V6","V7","V8","V9","class"]

Y=df["class"]
X=df[["name","V1","V2","V3","V4","V5","V6","V7","V8","V9"]]

Xtrain, Xtest, Ytrain, Ytest=train_test_split(X,Y, test_size=0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(Xtrain, Ytrain)

accuracy=clf.score(Xtest, Ytest)
accuracy


# # cLASIFICAR NUEVOS DATOS

# In[96]:


example=np.array([4,2,1,1,1,2,3,2,1])


# In[ ]:





# In[97]:


example=example.reshape(1,-1)


# In[ ]:





# In[98]:


example


# In[99]:


pred=clf.predict(example)
print(pred)


# In[102]:


sample_measure2=np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]]).reshape(2,-1)


# In[103]:


prediction=clf.predict(sample_measure2)


# In[104]:


prediction


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




