#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np


# In[7]:


emp_data = pd.read_csv('https://raw.githubusercontent.com/zekelabs/data-science-complete-tutorial/master/Data/HR_comma_sep.csv.txt')


# In[8]:


emp_data.head()


# In[9]:


emp_data.rename(columns={'sales':'department'}, inplace=True)


# In[10]:


emp_data.corr()['left']


# # Selecting categorical columns & integer columns

# In[11]:


cat_emp_data = emp_data.select_dtypes('object')


# In[12]:


int_emp_data = emp_data.select_dtypes('int64')


# # Preprocessing Categorical Columns

# In[13]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[14]:


le = LabelEncoder()
ohe = OneHotEncoder()


# In[15]:


le.fit(cat_emp_data.department)


# In[16]:


cat_emp_data['department_tf'] = le.transform(cat_emp_data.department)


# In[18]:


cat_emp_data.head()


# In[19]:


le.classes_


# In[20]:


le.inverse_transform([7])


# In[21]:


ohe.fit(cat_emp_data[['department_tf']])


# In[22]:


department_tf = ohe.transform(cat_emp_data[['department_tf']]).toarray()


# In[23]:


from sklearn.preprocessing import FunctionTransformer


# In[24]:


def func(x):
    def mapping(d):
        if d == 'low':
            return 1
        elif d == 'medium':
            return 2
        else:
            return 3
    return x.map( mapping )
        
ft = FunctionTransformer(func, validate=False)


# In[25]:


cat_emp_data['salary_tf'] = ft.transform(cat_emp_data.salary)


# # Preprocessing Number Data

# In[26]:


int_emp_data.head()


# In[27]:


int_emp_data.drop('left',axis=1, inplace=True)


# In[28]:


from sklearn.preprocessing import MinMaxScaler


# In[29]:


mms = MinMaxScaler()


# In[30]:


mms.fit(int_emp_data)


# In[31]:


int_tf = mms.transform(int_emp_data)
int_tf


# In[32]:


float_tf = emp_data[['satisfaction_level','last_evaluation']].values


# In[33]:


cat_emp_data['salary_tf'].values.shape


# In[34]:


int_tf.shape


# In[35]:


department_tf[:2]


# # Joining the data

# In[36]:


salary_tf = cat_emp_data['salary_tf'].values.reshape(-1,1)


# In[37]:


feature_data = np.hstack([department_tf,int_tf,float_tf,salary_tf])


# In[38]:


target_data = emp_data.left


# # Split data into two parts

# In[39]:


from sklearn.model_selection import train_test_split
trainX, testX, trainY, testY = train_test_split(feature_data,target_data)


# # Model Training

# In[40]:


from sklearn.linear_model import LogisticRegression


# In[41]:


from sklearn.ensemble import RandomForestClassifier
lr = LogisticRegression()
lr.fit(trainX,trainY)


# In[42]:


rf = RandomForestClassifier()
rf.fit(trainX,trainY)


# # Model Validation

# In[43]:


lr.score(testX,testY)


# In[44]:


rf.score(testX,testY)


# In[45]:


from sklearn.metrics import recall_score,precision_score, f1_score, classification_report


# In[46]:


pred = rf.predict(testX)


# In[47]:


precision_score(y_pred=pred, y_true=testY)


# In[48]:


f1_score(y_pred=pred, y_true=testY)

print (classification_report(y_pred=pred, y_true=testY))


# In[ ]:




