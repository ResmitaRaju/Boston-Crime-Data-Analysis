#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
data=pd.read_csv('H:\\2nd sem\\ML\\test.csv')
data


# In[32]:



#Contingency Table
contingency_table=pd.crosstab(data["Gender"],data["Like driving?"])
print('contingency_table :-\n',contingency_table)


# In[33]:



#Observed Values
Observed_Values = contingency_table.values 
print("Observed Values :-\n",Observed_Values)


# In[34]:


#Expected Values
import scipy.stats
b=scipy.stats.chi2_contingency(contingency_table)
Expected_Values = b[3]
print("Expected Values :-\n",Expected_Values)


# In[35]:


#Degree of Freedom
no_of_rows=len(contingency_table.iloc[0:2,0])
no_of_columns=len(contingency_table.iloc[0,0:2])
df=(no_of_rows-1)*(no_of_columns-1)
print("Degree of Freedom:-",df)

#or
#df=b[2]
#print("Degree of Freedom:-",df)


# In[36]:


#Significance Level 5%
alpha=0.05


# In[37]:


from scipy.stats import chi2
chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
chi_square_statistic=chi_square[0]+chi_square[1]
print("chi-square statistic:-",chi_square_statistic)


# In[38]:


#critical_value
critical_value=chi2.ppf(q=1-alpha,df=df)
print('critical_value:',critical_value)


# In[39]:


#p-value
p_value=1-chi2.cdf(x=chi_square_statistic,df=df)
print('p-value:',p_value)


# In[40]:


print('Significance level: ',alpha)
print('Degree of Freedom: ',df)
print('chi-square statistic:',chi_square_statistic)
print('critical_value:',critical_value)
print('p-value:',p_value)


# In[27]:


#compare chi_square_statistic with critical_value and p-value which is the probability of getting chi-square>0.09 (chi_square_statistic)
if chi_square_statistic>=critical_value:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")
    
if p_value<=alpha:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")


# CONCLUSION.
# 
# The dataset we have used is about the number of males and female who like to drive. It is a categorical dataset suitable for a chisquare test.
# The Chi-square test of independence tests if there is a relationship between two categorical variables. 
# The data is usually displayed in a cross-tabulation format with each row representing a level (group) 
# for one variable and each column representing a level (group) for another variable.
# The test is comparing the observed observations to the expected observations.
# 
# The Hypothesis are:
#     
#     The H0 (Null Hypothesis): There is no relationship between variable one and variable two.
# 
#     The H1 (Alternative Hypothesis): There is a relationship between variable 1 and variable 2.
#     
#     If the p-value is significant, you can reject the null hypothesis and 
#     claim that the findings support the alternative hypothesis.
#     
# We have obtained the following results 
#     Significance level:  0.05
#     Degree of Freedom:  1
#     chi-square statistic: 0.21978021978021978
#     critical_value: 3.841458820694124
#     p-value: 0.6392074309046059
#         
# Since p value and  is greater than 0.05 and chisquare value is lesser than critical value we accept the null hypothesis
# which states that there is no relationship between the genger and like driving columns.
