#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn import datasets
from io import StringIO
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report,confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#load data file
data = pd.read_excel(r"C:\Users\Lt col Haider\OneDrive\IBA\semester 4\Foundations to Data Science\project\default of credit card clients.xls")
data.head()


# In[3]:


data.shape


# # Data Cleaning

# In[4]:


#ID is unique variable so drop it
data =  data.drop(columns = ['ID'])


# In[5]:


#rename target column for ease 
data = data.rename(columns={'default payment next month': 'Default'})


# In[6]:


#categorical variables have already been dummified


# In[7]:


#check for missing values 
total = data.isnull().sum().sort_values(ascending = False)
percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()


# No missing data found.

# # Exploratory Data Analysis

# In[8]:


data.describe()


# In[9]:


temp = data["Default"].value_counts()
df = pd.DataFrame({'Default': temp.index,'values': temp.values})
plt.figure(figsize = (6,6))
plt.title('Default Credit Card Clients - target value - data unbalance\n (Not Default = 0, Default = 1)')
sns.set_color_codes("pastel")
sns.barplot(x = 'Default', y="values", data=df)
locs, labels = plt.xticks()
plt.show()


# data is imbalanced

# In[10]:


numerical = ['LIMIT_BAL','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3',
             'BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
categorical = ['SEX','EDUCATION','MARRIAGE']


# In[11]:


#T Test for numerical columns
p=[]
from scipy.stats import ttest_ind

for i in numerical:
    df1=data.groupby('Default').get_group(0)
    df2=data.groupby('Default').get_group(1)
    t,pvalue=ttest_ind(df1[i],df2[i])
    p.append(1-pvalue)
plt.figure(figsize=(7,7))
sns.barplot(x=p, y=numerical)
plt.title('Best Numerical Features')
plt.axvline(x=(1-0.05),color='r')
plt.xlabel('1-p value')
plt.show()


# All features show importance, and almost all of them show statistical significance so won't drop any.

# In[12]:


#Chi Square test for Categorical Columns
from scipy.stats import chi2_contingency
l=[]
for i in categorical:
    pvalue  = chi2_contingency(pd.crosstab(data['Default'],data[i]))[1]
    l.append(1-pvalue)
plt.figure(figsize=(7,7))
sns.barplot(x=l, y=categorical)
plt.title('Best Categorical Features')
plt.axvline(x=(1-0.05),color='r')
plt.show()


# All categorical features show statistical significance so won't drop any.

# In[13]:


#drop target variable: 'Default'
features =  data.drop(columns = ['Default'])


# In[14]:


#correlation analysis
corr_matrix = features.corr().abs()
corr_matrix


# In[15]:


#Feature Selection 

#Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]


# In[16]:


#Drop highly correlated attributes
data = data.drop(data[to_drop], axis=1)


# In[17]:


data.shape


# In[18]:


data.columns


# In[19]:


plt.figure(figsize=(6,6))
data.SEX.value_counts().plot(kind = 'bar')


# In[20]:


pd.crosstab(data.SEX, data.Default)


# Female population is in majority and is more likely to default on the loan. This could be explained by their high percentage in the dataset.

# In[21]:


plt.figure(figsize=(6,6))
data.EDUCATION.value_counts().plot(kind = 'bar')


# In[22]:


pd.crosstab(data.EDUCATION, data.Default)


# University education level in highest in the dataset, and is most likely to default. This could be explained by their high percentage in the dataset. Next are Graduates, and then Highschools. This should be the opposite, but again could be explained by the difference in their numbers in the dataset. Though, those least likely to default are unknown labels [0,4,5,6]. Logically speaking, they need to be above graduate level atleast as education level and likeliness of default has an inverse relationship. I tried removing these unknowns but they reduced the accuracy and AUC score. Thus, I believe that they carry important information. So I won't remove them.  

# In[23]:


plt.figure(figsize=(6,6))
data.MARRIAGE.value_counts().plot(kind = 'bar')


# In[24]:


pd.crosstab(data.MARRIAGE, data.Default)


# Single people are highest in number and are most likely to default. This makes sense. Type '3' may be people in a relationship. The '0' column is unknown. I tried removing them but this reduce model accuracy. Thus i infer that they carry important information and won't remove them.

# In[25]:


#density plot for amount of credit limit (LIMIT_BAL) grouped by default payment next month.
class_0 = data.loc[data['Default'] == 0]["LIMIT_BAL"]
class_1 = data.loc[data['Default'] == 1]["LIMIT_BAL"]
plt.figure(figsize = (14,6))
plt.title('Default amount of credit limit  - grouped by Payment Next Month (Density Plot)')
sns.set_color_codes("pastel")
sns.distplot(class_1,kde=True,bins=200, color="red")
sns.distplot(class_0,kde=True,bins=200, color="green")
plt.show()


# This shows that higher the credit limit, lower is the chance of default. This is sensible as richer people tend to have higher credit limit and are so less likely to default on loans. The highest defaulters are for credit limit 0 to 100,000, with the highest being for credit limit 50,000, and the density for this interval is larger for defaulters than for non-defaulters.

# In[26]:


def corr_2_cols(Col1, Col2):
    per = data.groupby([Col1, Col2]).size().unstack()
    per['perc'] = (per[per.columns[1]]/(per[per.columns[0]] + per[per.columns[1]]))
    return per

corr_2_cols('PAY_0', 'Default')


# No '-2' value qouted in data description. I am not able to infer it on inspection of the dataset too. Will keep it as I am not able to find an explanation.

# # Pre-Processing

# In[27]:


#Split test and train data
x = data.drop(columns = 'Default')
y = data['Default']

#will train on 70% of the data, test on the remaining 30%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=50)


# In[28]:


#feature scaling
#To avoid data Leakage, scale the x_train and x_test separately
scaler = RobustScaler(copy=True)
x_train = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train, columns = x.columns)

x_test = scaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test, columns = x.columns)


# In[29]:


#balancing the dataset
y.value_counts(normalize=True)


# Majority class (not default) has 77% samples. Minority class (default) has 22% samples.
# Data is highly imbalanced. Thus, will apply SMOTE and Undersampling techniques.

# In[30]:


from imblearn.over_sampling import SMOTE
#so as not to lose any important information, apply SMOTE (Synthetic Minority Oversampling Technique) to fix class imbalance
sm = SMOTE(random_state=10)
x_smote, y_smote = sm.fit_sample(x_train, y_train)


# In[31]:


#Undersampling data


# In[32]:


#create the training df by remerging x_train and y_train
df_train = x_train.join(y_train)
df_train.sample(10)


# In[33]:


from sklearn.utils import resample
# Separate majority and minority classes
df_majority = df_train[df_train.Default==0]
df_minority = df_train[df_train.Default==1]

print(df_majority.Default.count())
print("-----------")
print(df_minority.Default.count())
print("-----------")
print(df_train.Default.value_counts())


# In[34]:


# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=3402,     # to match minority class
                                 random_state=587) # reproducible results
# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])
# Display new class counts
df_downsampled.Default.value_counts()


# In[35]:


#Downsampled data
y_downsampled = df_downsampled.Default
x_downsampled = df_downsampled.drop(['Default'], axis = 1)


# # Modeling

# # 1. Logistic Regression

# In[36]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.001, 0.01, 1, 10]}

lr = LogisticRegression(penalty='l2', class_weight='balanced')
lr = GridSearchCV(estimator = lr, param_grid = param_grid, n_jobs = -1)
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[37]:


from sklearn.metrics import roc_curve, auc
y_prob = lr.predict_proba(x_test)[:,1]
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)

import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# In[38]:


lr.fit(x_smote, y_smote)
y_pred = lr.predict(x_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[39]:


lr.fit(x_downsampled, y_downsampled)
y_pred = lr.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Best result on original data = 0.69, AUC = 0.73

# # 2. Decision Tree

# In[40]:


from sklearn.tree import DecisionTreeClassifier

# Decision tree with depth = 2
dt2 = tree.DecisionTreeClassifier(random_state=42, max_depth=2)
dt2.fit(x_train, y_train)
dt2_score_train = dt2.score(x_train, y_train)
print("Training score: ",dt2_score_train)
dt2_score_test = dt2.score(x_test, y_test)
print("Testing score: ",dt2_score_test)


# In[41]:


# Decision tree with depth = 3
dt3 = tree.DecisionTreeClassifier(random_state=42, max_depth=3)
dt3.fit(x_train, y_train)
dt3_score_train = dt3.score(x_train, y_train)
print("Training score: ",dt3_score_train)
dt3_score_test = dt3.score(x_test, y_test)
print("Testing score: ",dt3_score_test)


# In[42]:


# Decision tree with depth = 4
dt4 = tree.DecisionTreeClassifier(random_state=42, max_depth=4)
dt4.fit(x_train, y_train)
dt4_score_train = dt4.score(x_train, y_train)
print("Training score: ",dt4_score_train)
dt4_score_test = dt4.score(x_test, y_test)
print("Testing score: ",dt4_score_test)


# In[43]:


# Decision tree: To the full depth
dt1 = tree.DecisionTreeClassifier()
dt1.fit(x_train, y_train)
dt1_score_train = dt1.score(x_train, y_train)
print("Training score: ", dt1_score_train)
dt1_score_test = dt1.score(x_test, y_test)
print("Testing score: ", dt1_score_test)


# In[44]:


#Compare Training and Testing scores for various tree depths used
print('{:10} {:20} {:20}'.format('depth', 'Training score','Testing score'))
print('{:10} {:20} {:20}'.format('-----', '--------------','-------------'))
print('{:1} {:>25} {:>20}'.format(2, dt2_score_train, dt2_score_test))
print('{:1} {:>25} {:>20}'.format(3, dt3_score_train, dt3_score_test))
print('{:1} {:>25} {:>20}'.format(4, dt4_score_train, dt4_score_test))
print('{:1} {:>23} {:>20}'.format("max", dt1_score_train, dt1_score_test))


# Highest testing accuracy reached at depth 4. At this depth, training accuracy is good too so will go with this.

# In[45]:


y_pred = dt4.predict(x_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[46]:


y_prob = dt4.predict_proba(x_test)[:,1]
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.figure(figsize=(6,6))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# In[47]:


dt4.fit(x_smote, y_smote)
dt4_score_train = dt4.score(x_smote, y_smote)
print("Training score: ",dt4_score_train)
dt4_score_test = dt4.score(x_test, y_test)
print("Testing score: ",dt4_score_test)

y_pred = dt4.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[48]:


dt4.fit(x_downsampled, y_downsampled)
dt4_score_train = dt4.score(x_downsampled, y_downsampled)
print("Training score: ",dt4_score_train)
dt4_score_test = dt4.score(x_test, y_test)
print("Testing score: ",dt4_score_test)

y_pred = dt4.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Best result on original data = 0.83, AUC = 0.75

# # 3. Random Forest Classifier 

# In[49]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
rf = RandomForestClassifier(n_estimators=100)


# In[50]:


rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[51]:


y_prob = rf.predict_proba(x_test)[:,1]
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.figure(figsize=(6,6))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# In[52]:


rf.fit(x_smote, y_smote)
y_pred = rf.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[54]:


rf.fit(x_downsampled, y_downsampled)
y_pred = rf.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Best result on original data = 0.83, AUC = 0.77

# # 4. Support Vector Machine (SVM)

# In[55]:


from sklearn import svm
from sklearn import metrics


# In[56]:


#Radial Basis Function Kernel with low gamma 0.01

clf = svm.SVC(kernel='rbf', random_state=0, gamma=.01, C=1) 

#Train the model using the training sets
clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))


# In[57]:


false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.figure(figsize=(6,6))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# In[58]:


clf = svm.SVC(kernel='rbf', random_state=0, gamma=.01, C=1) 

#Train the model using the training sets
clf.fit(x_smote, y_smote)

#Predict the response for test dataset
y_pred = clf.predict(x_test)

# model accuracy:
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[59]:


clf = svm.SVC(kernel='rbf', random_state=0, gamma=.01, C=1) 

#Train the model using the training sets
clf.fit(x_downsampled, y_downsampled)

#Predict the response for test dataset
y_pred = clf.predict(x_test)

# model accuracy:
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Best result on original data = 0.82, AUC = 0.64

# # 5. Naive Bayes

# In[60]:


from sklearn.naive_bayes import GaussianNB
naive = GaussianNB()
naive.fit(x_train, y_train)
    
# Make predictions and evalute
y_pred = naive.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))    
print(classification_report(y_test, y_pred))


# In[61]:


y_prob = naive.predict_proba(x_test)[:,1]
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.figure(figsize=(6,6))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# In[62]:


naive.fit(x_smote, y_smote)
    
# Make predictions and evalute
y_pred = naive.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))    
print(classification_report(y_test, y_pred))


# In[63]:


naive.fit(x_downsampled, y_downsampled)
y_pred = naive.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Best result on original data = 0.58, AUC = 0.74

# # 6. XGBoost Classifier

# In[64]:


from xgboost import XGBClassifier

model = XGBClassifier(silent=False, 
                      scale_pos_weight=1,
                      learning_rate=0.01,  
                      colsample_bytree = 0.8,
                      subsample = 0.8,
                      objective='binary:logistic', 
                      n_estimators=100, 
                      reg_alpha = 0.3,
                      max_depth=4, 
                      gamma=5)


# In[65]:


model.fit(x_train, y_train)
    
# Make predictions and evalute
y_pred = model.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred)) 
print(classification_report(y_test, y_pred))


# In[66]:


y_prob = model.predict_proba(x_test)[:,1]
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.figure(figsize=(6,6))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# In[67]:


model.fit(x_smote, y_smote)
    
# Make predictions and evalute
y_pred = model.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred)) 
print(classification_report(y_test, y_pred))


# In[68]:


model.fit(x_downsampled, y_downsampled)
y_pred = model.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Best result on original data = 0.83, AUC = 0.78

# Best Accuracy = 0.83, DT, RF, XGB |  Best AUC = 0.78 XGB | Best Model = XGBoost Classifier
