#!/usr/bin/env python
# coding: utf-8

# # Data Import

# In[39]:


# Import Library
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Declare File Directory
Directory = r"C:\Users\User\Documents"

data = pd.read_csv(f"{Directory}\Thyroid_Disease.csv", delimiter=';')


# # Data Exploration

# In[40]:


data.head()


# ## Duplicate Values

# In[41]:


# Check duplicate rows
data[data.duplicated()]


# In[42]:


# Remove duplicate rows
data.drop_duplicates(keep='first', inplace=True)


# In[43]:


# check any duplicate rows left
data[data.duplicated()]


# ## Attribute Data Type

# In[44]:


data.columns


# In[45]:


data.dtypes


# ## Drop Unnecessary Columns

# In[46]:


data = data.drop(["Unnamed: 22", "Unnamed: 23"], axis=1)
data.columns


# ## Statistical

# In[47]:


data.shape


# In[48]:


data.describe()


# In[49]:


data.info()


# In[50]:


data['Outlier_label '].value_counts()


# ## Missing Values

# In[51]:


data.isnull().sum()


# # Data Cleaning

# ## Data Encoding

# In[52]:


data['Outlier_label '] = data['Outlier_label '].astype("category").cat.codes


# In[53]:


data.dtypes


# ## Rename Columns

# In[54]:


data.columns = data.columns.str.replace(' ', '')


# ## Feature Scaling

# In[55]:


from sklearn.preprocessing import MinMaxScaler

for col in data.select_dtypes(include=[float]).columns:
    data[col] = MinMaxScaler().fit_transform(
        np.array(data[col]).reshape(-1, 1))


# ## Data Splitting

# In[56]:


X_train = data.drop(['Outlier_label'], axis=1)
Y_train = data['Outlier_label']


# # Data Visualization

# ## Correlation Matrix

# In[57]:


corr = data.apply(lambda x: pd.factorize(x)[0]).corr(method='pearson',
                                                     min_periods=1)
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# ## Bar Chart

# In[58]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.set(rc={'figure.figsize':(5,5)})

category = ['Sex', 'on_thyroxine', 'query_on_thyroxine',
       'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
       'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium',
       'goitre', 'tumor', 'hypopituitary', 'psych']

for col in category:
    sns.countplot(x=col, hue='Outlier_label', data=data)
    plt.title('Bar Chart', fontsize=20)
    plt.xlabel(col, fontsize=15)
    plt.ylabel('Count', fontsize=15)
    plt.show()


# ## Box Plot

# In[59]:


sns.set(rc={'figure.figsize':(10,5)})

sns.boxplot(data=data[['Age', 'TSH', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured']])
plt.title('Boxplot', fontsize=20)
plt.xlabel('Features', fontsize=16)
plt.ylabel('Values', fontsize=16)


# ## Pair Plot

# In[60]:


graph = sns.pairplot(data,
             kind='scatter',
             diag_kind='hist',
             hue='Outlier_label',
             x_vars=data[['TSH', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured']],
             y_vars='Outlier_label')
graph.fig.set_size_inches(20, 5)


# # Data Modelling

# In[61]:


import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import RocCurveDisplay, auc, roc_curve, roc_auc_score, accuracy_score, classification_report

#random state
rs = 10
outlier_fraction = 0.1


# ## Isolation Forest

# In[62]:


IForest = IsolationForest(random_state=10, contamination=outlier_fraction)

# fit the dataset to the model
IForest.fit(X_train)

# prediction of a datapoint category outlier or inlier
y_pred = IForest.predict(X_train)

y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1

# no of errors in prediction
n_errors = (y_pred != Y_train).sum()
print("Model:  Isolation Forest")
print("No of Errors:", n_errors)
print("Accuracy:", accuracy_score(Y_train, y_pred))
print(classification_report(Y_train, y_pred, digits=4))

y_pred_proba = IForest.decision_function(X_train)
y_pred_proba = [-1*s + 0.5 for s in y_pred_proba]

print("AUC Score: ", roc_auc_score(Y_train, y_pred_proba))
fpr, tpr, _ = roc_curve(Y_train, y_pred_proba)
roc_auc = auc(fpr, tpr)
rcd = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
rcd.plot()
plt.title("Isolation Forest ROC Curve")
plt.show()


# ## LocalOutlierFactor

# In[63]:


LOF = LocalOutlierFactor(novelty=True, contamination=outlier_fraction)


# fit the dataset to the model
LOF.fit(X_train)

# prediction of a datapoint category outlier or inlier
y_pred = LOF.predict(X_train)

y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1

# no of errors in prediction
n_errors = (y_pred != Y_train).sum()
print("Model:  LocalOutlierFactor")
print("No of Errors:", n_errors)
print("Accuracy:", accuracy_score(Y_train, y_pred))
print(classification_report(Y_train, y_pred, digits=4))

y_pred_proba = LOF.decision_function(X_train)
y_pred_proba = [-1*s + 0.5 for s in y_pred_proba]

print("AUC Score: ", roc_auc_score(Y_train, y_pred_proba))
fpr, tpr, _ = roc_curve(Y_train, y_pred_proba)
roc_auc = auc(fpr, tpr)
rcd = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
rcd.plot()
plt.title("LOF ROC Curve")
plt.show()


# ## OneClassSVM

# In[64]:


OCSVM = OneClassSVM(nu=outlier_fraction)

# prediction of a datapoint category outlier or inlier
y_pred = OCSVM.fit_predict(X_train)

y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1

# no of errors in prediction
n_errors = (y_pred != Y_train).sum()
print("Model:  OneClassSVM")
print("No of Errors:", n_errors)
print("Accuracy:", accuracy_score(Y_train, y_pred))
print(classification_report(Y_train, y_pred, digits=4))

y_pred_proba = OCSVM.decision_function(X_train)
y_pred_proba = MinMaxScaler().fit_transform(y_pred_proba.reshape(-1, 1))

print("AUC Score: ", roc_auc_score(Y_train, y_pred_proba))
fpr, tpr, _ = roc_curve(Y_train, y_pred_proba)
roc_auc = auc(fpr, tpr)
rcd = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
rcd.plot()
plt.title("OCSVM ROC Curve")
plt.show()


# ## KNearestNeighbors

# In[89]:


KNN = KNeighborsClassifier()
KNN.fit(X_train, Y_train)

X_train['Outlier_probability'] = KNN.predict_proba(X_train)[:,1]

# if probobability >= 0.5, then = outlier
def prediction(value):
    if value >= 0.5:
        return 1
    else:
        return 0
    
X_train['Outlier_label'] = X_train['Outlier_probability'].map(prediction)

n_errors = (X_train['Outlier_label'] != Y_train).sum()
print("Model:  KNeighborsClassifier")
print("No of Errors:", n_errors)
print("Accuracy:", accuracy_score(Y_train, X_train['Outlier_label']))
print(classification_report(Y_train, X_train['Outlier_label'], digits=4))

print("AUC Score: ", roc_auc_score(Y_train, X_train['Outlier_probability']))
fpr, tpr, thresholds = roc_curve(Y_train, X_train['Outlier_probability'])
roc_auc = auc(fpr, tpr)
rcd = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
rcd.plot()
plt.title("KNN ROC Curve")
plt.show()


# ## Model Tuning - KNN 

# In[92]:


KNN.get_params()


# In[93]:


from sklearn.model_selection import GridSearchCV

params = {
    'n_neighbors': [5, 10, 20, 30],
    'leaf_size': [10, 20, 30, 40, 50],
    'p': [1, 2]
}

KNN_tuned = GridSearchCV(KNeighborsClassifier(), params, cv=10)

KNN_tuned.fit(X_train, Y_train)

print("Best: %f using %s" % (KNN_tuned.best_score_, KNN_tuned.best_params_))


# ## SHAP - KNN

# In[32]:


data_subset = data.sample(n=1000, random_state = 20)

X_train_subset = data_subset.drop(['Outlier_label'], axis=1)
Y_train_subset = data_subset['Outlier_label']


# In[33]:


import shap

KNN = KNeighborsClassifier(leaf_size=10, n_neighbors=20, p=1)

# fit the dataset to the model
KNN.fit(X_train_subset, Y_train_subset)

exp = shap.SamplingExplainer(KNN.predict, X_train_subset)  #Explainer
shap_values = exp.shap_values(X_train_subset)  #Calculate SHAP values
shap.initjs()  #load JS visualization code to notebook


# In[34]:


shap.summary_plot(shap_values, X_train_subset, plot_type="bar")


# In[35]:


shap.summary_plot(shap_values, X_train_subset)


# # Data Export

# In[94]:


import datetime
import os
     
# get current date
date = datetime.datetime.now()

# check folder exist
if not os.path.exists(f"{Directory}\Thyroid_Prediction_{date.year}"):
    os.makedirs(f"{Directory}\Thyroid_Prediction_{date.year}")

#Extract Outlier  
Outlier = X_train.loc[X_train['Outlier_label'] == 1]

# save file
Outlier.to_csv(f"{Directory}\Thyroid_Prediction_{date.year}\Thyroid_Prediction_{date.day}.csv", index = False)

