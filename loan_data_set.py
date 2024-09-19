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
data = pd.read_csv(r"C:\Users\Lt col Haider\OneDrive\IBA\semester 4\Foundations to Data Science\project\loan_data_set.csv")
data.head()


# In[3]:


data.shape


# # Data Cleaning

# In[4]:


#Loan_ID is a unique variable so drop it
data =  data.drop(columns = ['Loan_ID'])


# In[5]:


#re-label categorical variables
data.Loan_Status.replace({'Y': 0, 'N':1}, inplace=True)
data.Gender.replace({'Male':1, 'Female':0}, inplace=True)
data.Married.replace({'Yes': 1, 'No':0}, inplace=True)
data.Education.replace({'Graduate': 1, 'Not Graduate':0}, inplace=True)
data.Self_Employed.replace({'Yes': 1, 'No':0}, inplace=True)


# In[6]:


#create dummy variables for 'Property_Area' and 'Dependents'
data = data.join(pd.get_dummies(data.Dependents, prefix='Dependents'))
data.drop(columns= ['Dependents'], inplace=True)
data = data.join(pd.get_dummies(data.Property_Area, prefix='Property_Area'))
data.drop(columns= ['Property_Area'], inplace=True)


# In[7]:


data.head()


# In[8]:


#check for missing values 
total = data.isnull().sum().sort_values(ascending = False)
percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()


# In[9]:


#Normalize data for effective KNN Imputation
#will manually normalize as NaN values present so preprocessing can't be used
def normalize(data):
    return (data - data.min()) * 1.0 / (data.max() - data.min())
data = data.apply(normalize)


# In[10]:


#Apply KNN Imputation
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=15)
df = imputer.fit_transform(data)


# In[11]:


#convert array back to dataframe
data = pd.DataFrame(df)
#bring data back in original form
data.columns = ['Gender','Married','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount',
                'Loan_Amount_Term','Credit_History','Loan_Status','Dependents_0','Dependents_1','Dependents_2','Dependents_3+',
                'Property_Area_Rural','Property_Area_Semiurban','Property_Area_Urban']
data.head()


# In[12]:


#check for missing values again
total = data.isnull().sum().sort_values(ascending = False)
percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()


# In[13]:


#check outliers in dataset using Z-Score
from scipy import stats
z = np.abs(stats.zscore(data))
print(z)                            
threshold = 3
print(np.where(z > 3))


# In[14]:


#remove outliers
data = data[(z < 3).all(axis=1)]
data.shape


# # Exploratory Data Analysis

# In[15]:


data.describe()


# In[16]:


temp = data["Loan_Status"].value_counts()
df = pd.DataFrame({'Loan_Status': temp.index,'values': temp.values})
plt.figure(figsize = (6,6))
plt.title('Default Clients - target value - data unbalance\n (Not Default = 0, Default = 1)')
sns.set_color_codes("pastel")
sns.barplot(x = 'Loan_Status', y="values", data=df)
locs, labels = plt.xticks()
plt.show()


# Data is imbalanced.

# In[17]:


numerical = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']
categorical = ['Credit_History','Gender','Married','Education','Self_Employed','Dependents_0','Dependents_1','Dependents_2',
               'Dependents_3+','Property_Area_Rural','Property_Area_Semiurban','Property_Area_Urban']


# In[18]:


#T Test for numerical columns
p=[]
from scipy.stats import ttest_ind

for i in numerical:
    df1=data.groupby('Loan_Status').get_group(0)
    df2=data.groupby('Loan_Status').get_group(1)
    t,pvalue=ttest_ind(df1[i],df2[i])
    p.append(1-pvalue)
plt.figure(figsize=(7,7))
sns.barplot(x=p, y=numerical)
plt.title('Best Numerical Features')
plt.axvline(x=(1-0.05),color='r')
plt.xlabel('1-p value')
plt.show()


# 'CoappliantIncome' has the greatest importance. None of them show statistical significance though.

# In[19]:


#Chi Square test for Categorical Columns
from scipy.stats import chi2_contingency
l=[]
for i in categorical:
    pvalue  = chi2_contingency(pd.crosstab(data['Loan_Status'],data[i]))[1]
    l.append(1-pvalue)
plt.figure(figsize=(7,7))
sns.barplot(x=l, y=categorical)
plt.title('Best Categorical Features')
plt.axvline(x=(1-0.05),color='r')
plt.show()


# 'Credit_History', 'Property_Area_Rural' and 'Property_Area_Semiurban' show statistical significance. 'Dependents_3+' has very low significance, so drop it.

# In[20]:


data =  data.drop(columns = ['Dependents_3+'])


# In[21]:


#drop target variable: 'Default'
features =  data.drop(columns = ['Loan_Status'])


# In[22]:


#correlation analysis
corr_matrix = features.corr().abs()
corr_matrix


# In[23]:


#Feature Selection 

#Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
#Drop highly correlated attributes
data = data.drop(data[to_drop], axis=1)
data.shape


# No highly correlated attributes found.

# In[24]:


pd.crosstab(data.Married, data.Loan_Status)


# Those not married have a higher percentage on defaulting.

# In[25]:


pd.crosstab(data.Education, data.Loan_Status)


# Graduates are more in number and thus are more in number for both default and not defaut. Not Graduates have a higher percentage of defaulting which makes sense. 

# In[26]:


pd.crosstab(data.Dependents_0, data.Loan_Status)


# Percentage of default for those with no dependents and those with dependents is almost the same.

# In[27]:


class_0 = data.loc[data['Loan_Status'] == 0]["ApplicantIncome"]
class_1 = data.loc[data['Loan_Status'] == 1]["ApplicantIncome"]
plt.figure(figsize = (14,6))
plt.title('Default amount (Density Plot)')
sns.set_color_codes("pastel")
sns.distplot(class_1,kde=True,bins=200, color="red")
sns.distplot(class_0,kde=True,bins=200, color="green")
plt.show()


# Data is right skewed. Those with less Applicant Incomes have higher defaulting tendency. The highest default is for value 0.04.

# In[28]:


class_0 = data.loc[data['Loan_Status'] == 0]["LoanAmount"]
class_1 = data.loc[data['Loan_Status'] == 1]["LoanAmount"]
plt.figure(figsize = (14,6))
plt.title('Default amount (Density Plot)')
sns.set_color_codes("pastel")
sns.distplot(class_1,kde=True,bins=200, color="red")
sns.distplot(class_0,kde=True,bins=200, color="green")
plt.show()


# Data has a normal distribution.

# # Pre-Processing

# In[29]:


#Split test and train data
x = data.drop(columns = 'Loan_Status')
y = data['Loan_Status']

#will train on 70% of the data, test on the remaining 30%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=50)


# In[30]:


y.value_counts(normalize=True)


# Majority class (not default) has 69% samples. Minority class (default) has 31% samples. Data is imbalanced. Thus, will apply SMOTE.

# In[31]:


from imblearn.over_sampling import SMOTE
#so as not to lose any important information, apply SMOTE (Synthetic Minority Oversampling Technique) to fix class imbalance
sm = SMOTE(random_state=10)
x_smote, y_smote = sm.fit_sample(x_train, y_train)


# # Modeling

# # 1. Logistic Regression

# In[32]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.001, 0.01, 1, 10]}

lr = LogisticRegression(penalty='l2', class_weight='balanced')
lr = GridSearchCV(estimator = lr, param_grid = param_grid, n_jobs = -1)
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[33]:


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


# In[34]:


lr.fit(x_smote, y_smote)
y_pred = lr.predict(x_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Best result on original data = 0.82, AUC = 0.88

# # 2. Decision Tree

# In[35]:


from sklearn.tree import DecisionTreeClassifier

# Decision tree with depth = 2
dt2 = tree.DecisionTreeClassifier(random_state=42, max_depth=2)
dt2.fit(x_train, y_train)
dt2_score_train = dt2.score(x_train, y_train)
print("Training score: ",dt2_score_train)
dt2_score_test = dt2.score(x_test, y_test)
print("Testing score: ",dt2_score_test)


# In[36]:


# Decision tree with depth = 3
dt3 = tree.DecisionTreeClassifier(random_state=42, max_depth=3)
dt3.fit(x_train, y_train)
dt3_score_train = dt3.score(x_train, y_train)
print("Training score: ",dt3_score_train)
dt3_score_test = dt3.score(x_test, y_test)
print("Testing score: ",dt3_score_test)


# In[37]:


# Decision tree with depth = 4
dt4 = tree.DecisionTreeClassifier(random_state=42, max_depth=4)
dt4.fit(x_train, y_train)
dt4_score_train = dt4.score(x_train, y_train)
print("Training score: ",dt4_score_train)
dt4_score_test = dt4.score(x_test, y_test)
print("Testing score: ",dt4_score_test)


# In[38]:


# Decision tree: To the full depth
dt1 = tree.DecisionTreeClassifier()
dt1.fit(x_train, y_train)
dt1_score_train = dt1.score(x_train, y_train)
print("Training score: ", dt1_score_train)
dt1_score_test = dt1.score(x_test, y_test)
print("Testing score: ", dt1_score_test)


# In[39]:


#Compare Training and Testing scores for various tree depths used
print('{:10} {:20} {:20}'.format('depth', 'Training score','Testing score'))
print('{:10} {:20} {:20}'.format('-----', '--------------','-------------'))
print('{:1} {:>25} {:>20}'.format(2, dt2_score_train, dt2_score_test))
print('{:1} {:>25} {:>20}'.format(3, dt3_score_train, dt3_score_test))
print('{:1} {:>25} {:>20}'.format(4, dt4_score_train, dt4_score_test))
print('{:1} {:>23} {:>20}'.format("max", dt1_score_train, dt1_score_test))


# Highest testing accuracy reached at depth 2. At this depth, training accuracy is good too so will go with this.

# In[40]:


y_pred = dt2.predict(x_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[41]:


y_prob = dt2.predict_proba(x_test)[:,1]
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


# In[42]:


dt2.fit(x_smote, y_smote)
dt2_score_train = dt2.score(x_smote, y_smote)
print("Training score: ",dt2_score_train)
dt2_score_test = dt2.score(x_test, y_test)
print("Testing score: ",dt2_score_test)

y_pred = dt2.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Best result on original data = 0.85, AUC = 0.82

# # 3. Random Forest Classifier

# In[43]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
rf = RandomForestClassifier(n_estimators=100)


# In[44]:


rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[45]:


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


# In[46]:


rf.fit(x_smote, y_smote)
y_pred = rf.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Best result on original data = 0.86, AUC = 0.86

# # 4. Support Vector Machine (SVM)

# In[47]:


from sklearn import svm
from sklearn import metrics


# In[48]:


#Radial Basis Function Kernel with low gamma 0.01

clf = svm.SVC(kernel='rbf', random_state=0, gamma=.01, C=1) 

#Train the model using the training sets
clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))


# In[49]:


#Radial Basis Fnction Kernel with penalty parameter c = 1000.
#classifier starts to become very intolerant to misclassified data points and thus the decision boundary becomes less biased and has more variance (i.e. more dependent on the individual data points)
clf = svm.SVC(kernel='rbf', random_state=0, gamma=.01, C=1000) 

#Train the model using the training sets
clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)

#takes too much training time.
from sklearn import metrics

# model accuracy:
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[50]:


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


# In[51]:


#Radial Basis Fnction Kernel with penalty parameter c = 1000.
#classifier starts to become very intolerant to misclassified data points and thus the decision boundary becomes less biased and has more variance (i.e. more dependent on the individual data points)
clf = svm.SVC(kernel='rbf', random_state=0, gamma=.01, C=1000) 

#Train the model using the training sets
clf.fit(x_smote, y_smote)

#Predict the response for test dataset
y_pred = clf.predict(x_test)

#takes too much training time.
from sklearn import metrics

# model accuracy:
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Best result on original data = 0.86, AUC = 0.78

# # 5. Naive Bayes

# In[52]:


from sklearn.naive_bayes import GaussianNB
naive = GaussianNB()
naive.fit(x_train, y_train)
    
# Make predictions and evalute
y_pred = naive.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))    
print(classification_report(y_test, y_pred))


# In[53]:


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


# In[54]:


naive.fit(x_smote, y_smote)
    
# Make predictions and evalute
y_pred = naive.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))    
print(classification_report(y_test, y_pred))


# Best result on original data = 0.86, AUC = 0.88

# # 6. XGBoost Classifier

# In[55]:


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


# In[56]:


model.fit(x_train, y_train)
    
# Make predictions and evalute
y_pred = model.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred)) 
print(classification_report(y_test, y_pred))


# In[57]:


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


# In[58]:


model.fit(x_smote, y_smote)
    
# Make predictions and evalute
y_pred = model.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred)) 
print(classification_report(y_test, y_pred))


# In[59]:


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


# Best result on original data = 0.87, AUC = 0.89

# In[ ]:


Best Accuracy = 0.87, XGB | Best AUC = 0.89 XGB | Best Model = XGBoost Classifier

