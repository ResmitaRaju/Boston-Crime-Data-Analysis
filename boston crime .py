#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import norm

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats
from IPython.display import display, HTML


# In[ ]:


data = pd.read_csv("C://Users//RENJITA//Downloads//crime.csv//crime.csv",encoding = "ISO-8859-1")


# The boston dataset is being loaded into jupiter notebook with the help of panda library.

# In[44]:


data.describe


# In[46]:



data.head(10)


# In[64]:


data.shape


# In[30]:


data.dtypes


# In[23]:


#year-wise crime
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
sns.countplot(x="YEAR", data=data)
ax.set_title("No. of crimes occurred in the year")
plt.tight_layout()
plt.show()


# In[35]:


ax = sns.barplot(x="STREET", y="REPORTING_AREA", data = streets)
ax.set(xlabel='Crime Street', ylabel='# of Crimes reported')
ax.set_xticklabels(streets['STREET'],rotation=90)

plt.show()


# In[38]:


# Top 10 Offense types

streets = data.groupby([data['OFFENSE_CODE_GROUP']])['STREET'].aggregate(np.size).reset_index().sort_values('STREET', ascending = False).head(10)
streets


# In[25]:


#District-wise shooting occurred
shootingOccurred = data[data["SHOOTING"] == "Y"].groupby("DISTRICT").agg("SHOOTING").count().reset_index().sort_values("SHOOTING", ascending=False)
shootingOccurred
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
p = sns.barplot(x=shootingOccurred.DISTRICT, y=shootingOccurred.SHOOTING, data=shootingOccurred)
plt.tight_layout()
plt.show()


# In[26]:


#Top 10 most crime occurred street
df = data["STREET"].value_counts().head(10).reset_index()
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111)
p = sns.barplot(x=df.iloc[:,0], y=df["STREET"], data=df)#, palette="spring")
p.set_ylabel("No. of Crimes Occurred")
p.set_xlabel("Street Name")
p.set_xticklabels(p.get_xticklabels(), rotation=90)
plt.tight_layout()
plt.show()


# In[27]:


#No. of hours spent on each Offence by district
df = data.groupby(["DISTRICT", "OFFENSE_CODE_GROUP"])["HOUR"].sum().reset_index().sort_values("HOUR", ascending=False)
df
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111)
p = sns.scatterplot(x="HOUR", y="OFFENSE_CODE_GROUP", hue="DISTRICT", data=df, palette="summer")
p.set_ylabel("No. of Crimes Occurred")
p.set_xlabel("Hours")
plt.tight_layout()
plt.show()


# In[41]:


#Year wise percentage rate
yrlbl = data['YEAR'].astype('category').cat.categories.tolist()
yrwisecount = data['YEAR'].value_counts()
sizes = [yrwisecount[year] for year in yrlbl]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=yrlbl,  autopct='%1.1f%%',shadow=True) 
ax1.axis('equal')
ax1.set_title("Year wise crime %")
plt.show()


# In[42]:


# Day of the week crime %
dayofwkcount = data['DAY_OF_WEEK'].value_counts()
dayofwkcount

dayofwk = data['DAY_OF_WEEK'].astype('category').cat.categories.tolist()
dayofwk

sizes = [dayofwkcount[dow] for dow in dayofwk]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=dayofwk,  autopct='%1.1f%%',shadow=True) 
ax1.axis('equal')
ax1.set_title("Day of week crime %")
plt.show()


# In[48]:


data.corr()


# In[50]:


sns.heatmap(data.corr(), vmax=1., square=False).xaxis.tick_top()


# In[52]:


data.groupby('MONTH').size().plot(kind="bar")


# In[55]:


data.groupby('HOUR').size().plot(kind = 'bar')


# In[63]:


sns.FacetGrid(data,hue="MONTH",size=5).map(sns.distplot,"HOUR").add_legend()


# Crime incident reports are provided by Boston Police Department (BPD) to document the
# initial details surrounding an incident to which BPD offers respond. This is a dataset containing
# records from the new  cirme incident report system,which includes a reduced set of feilds focused on capturing the type of
# incident as well as when and where it occured. Recods starts from 2015-2018.
# In the above diagrams it is seen that
#    :the greatest number of crime occured is in the year of 2017.
#    :Highest crime is reported in Washigton Street.
#    :the top 10 crimes are 
#     Motor Vehicle Accident Response, Larceny,Medical Assistance, Investigate Person, Other,
#     Drug Violation, Simple Assault, Vandalism, Verbal Disputes, and Towed.
#    :District B2 has the heihest crime rate whereas A15 has the lowest crime rate.
#    :According to Year wise crime percentage rate  2017 shows the heighest percentage rate but it is to be noted that in the next here the 
#     crime rate percentage has gone down drastically.
#    :If we consider the Day of the week crime percentage rate it is to be noted that most of the crime is occuring on fridays.
#    :On estimating the correlation between variables it is to be noted that latitude and logitude are heihly correlated with value 0.99.
#    : A correlation matrix is also obtained for better visualization.
#    :on Comparing the hour and month of the crimes it is to be noted that it is following almost the same distribution.
