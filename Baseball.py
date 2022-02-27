#!/usr/bin/env python
# coding: utf-8

# Problem Statement:
# This dataset utilizes data from 2014 Major League Baseball seasons in order to develop an algorithm that predicts the number of wins for a given team in the 2015 season based on several different indicators of success. There are 16 different features that will be used as the inputs to the machine learning and the output will be a value that represents the number of wins. 

# Input features: Runs, At Bats, Hits, 
# Doubles, Triples, Homeruns, Walks, 
# Strikeouts, Stolen Bases,
# Runs Allowed, Earned Runs, Earned Run Average
# (ERA), Shutouts, Saves, Complete Games 
# and Errors
# -- Output: Number of predicted wins (W)
# To understand the columns meaning, follow the link given below to understand the baseball statistics: https://en.wikipedia.org/wiki/Baseball_statistics

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

url='https://raw.githubusercontent.com/dsrscientist/Data-Science-ML-Capstone-Projects/master/baseball.csv'
df=pd.read_csv(url)


# In[2]:


df.describe()


# In[3]:


df.shape


# In[4]:


df.dtypes


# In[5]:


df.head(10)


# In[6]:


# to check if null value exists in data
# df.apply(lambda x: sum(x.isnull()))
sns.heatmap(df.isnull().sum().to_frame())


# Shows no null values are present.

# In[7]:


# plotting correlation heatmap
dataplot = sns.heatmap(df.corr(), cmap="YlGnBu")
  
# displaying heatmap
plt.show()


# RA, ER & ERA have strong negative corelation with wins.
# SV have strong positive corelation with wins.
RA, ER & ERA look inter-related.
# In[8]:


def pplot(df,i):
    plt.figure(figsize=(20,5),facecolor='white')
    plt.subplot(1,3,1)
    sns.histplot(x=i,data=df,kde=True)
    plt.subplot(1,3,2)
    sns.regplot(x=i,y='W',data=df)
    plt.subplot(1,3,3)
    sns.boxplot(y=i,data=df)
    plt.show()
plotnum=1
for column in df:
    if(plotnum<=17):
        pplot(df,column)
    plotnum+=1
plt.tight_layout()


# In[9]:


x = df.drop(columns=['W'])
y = df['W']


# In[10]:


plt.figure(figsize=(20,20),facecolor='white')
plotnum=1
for column in x:
    if(plotnum<=16):
        ax=plt.subplot(4,4,plotnum)
        plt.scatter(x[column],y)
        plt.xlabel(column,fontsize=16)
        plt.ylabel('WINS',fontsize=16)

    plotnum+=1
plt.tight_layout()


# In[117]:


sns.regplot(x='SV',y='W',data=df)
plt.show()


# In[11]:


sns.regplot(x='ER',y='W',data=df)


# In[14]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[32]:


x_scaled=scaler.fit_transform(x)


# In[33]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( x_scaled, y, test_size=0.3)


# In[19]:


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
 
lm.fit(X_train, y_train)


# In[34]:


y_pred=lm.predict(X_test)
# plotting predictions
plt.figure(figsize=(10,5))
plt.scatter(x_test,y_test,s=15)
plt.plot(x_test,y_pred,color='r')
plt.xlabel('Predictor',fontsize=16)
plt.ylabel('Target',fontsize=16)
plt.show()


# In[22]:


lm.score(X_train,y_train)


# In[23]:


lm.score(X_test,y_test)


# In[24]:


print('Coefficients: \n', lm.coef_)


# In[25]:


from sklearn.metrics import mean_squared_error, r2_score
r2_score(y_test,y_pred)


# In[26]:


print("Mean squared error: %.2f" % mean_squared_error(y_test,y_pred))


# In[27]:


# Generate plot
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')
 
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[28]:


# Putting together the test and predicted values
result = pd.DataFrame()
result["y_test"] = y_test
result['prediction'] = y_pred

result

LASSO REGULARIZATION
# In[30]:


from sklearn.linear_model import Lasso, Ridge, LassoCV,RidgeCV
from sklearn.model_selection import KFold, RepeatedKFold, GridSearchCV, cross_validate

lassCv=LassoCV(alphas=None,cv=10,max_iter=20000,normalize=True)
lassCv.fit(X_train,y_train)


# In[31]:


alpha=lassCv.alpha_
alpha


# In[196]:


lasso_reg=Lasso(alpha)
lasso_reg.fit(X_train,y_train)


# In[197]:


lasso_reg.score(X_test,y_test)


# In[198]:


lass_pred=lasso_reg.predict(X_test)


# In[263]:


# plotting predictions

plt.figure(figsize=(7,7))
plt.scatter(y_test, lass_pred, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(lass_pred), max(y_test))
p2 = min(min(lass_pred), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()


# In[199]:


# Putting together the test and predicted values
result = pd.DataFrame()
result["y_test"] = y_test
result['lasso_prediction'] = lass_pred

result

RIDGE REGULARIZATION
# In[176]:


#ridge
alphas=np.random.uniform(low=0,high=10,size=(50,))
ridgeCv=RidgeCV(alphas=alphas,cv=10,normalize=True)
ridgeCv.fit(X_train,y_train)


# In[177]:


ridgeCv.alpha_


# In[178]:


ridgeModel=Ridge(alpha=ridgeCv.alpha_)
ridgeModel.fit(X_train,y_train)


# In[179]:


ridgeModel.score(X_test,y_test)


# In[180]:


ridge_pred=ridgeModel.predict(X_test)
# Putting together the test and predicted values
result = pd.DataFrame()
result["y_test"] = y_test
result['ridge_prediction'] = ridge_pred

result


# In[266]:


# plotting predictions

plt.figure(figsize=(7,7))
plt.scatter(y_test, ridge_pred, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(ridge_pred), max(y_test))
p2 = min(min(ridge_pred), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()


# In[221]:


# importing libraries for polynomial transform
from sklearn.preprocessing import PolynomialFeatures
# for creating pipeline
from sklearn.pipeline import Pipeline
# creating pipeline and fitting it on data
Input=[('polynomial',PolynomialFeatures(degree=2)),('modal',LinearRegression())]
pipe=Pipeline(Input)


# In[230]:


x_cols=df.drop(columns=['W'])
x_cols.columns


# In[231]:


X_train_df = pd.DataFrame(X_train, columns =x_cols.columns)


# In[225]:


y_train_df = pd.DataFrame()
y_train_df['y_train'] = y_train


# In[232]:


pipe.fit(X_train_df,y_train_df)


# In[233]:


poly_pred=pipe.predict(X_test)


# In[234]:


# Putting together the test and predicted values
poly_result = pd.DataFrame()
poly_result["y_test"] = y_test
poly_result['poly_pred'] = poly_pred

poly_result

VARIANCE INFLATION FACTOR
# In[240]:


plt.scatter(df['ER'],df['RA'])
plt.xlabel('ER',fontsize=16)
plt.ylabel('RA',fontsize=16)
plt.show()


# In[241]:


plt.scatter(df['ER'],df['ERA'])
plt.xlabel('ER',fontsize=16)
plt.ylabel('ERA',fontsize=16)
plt.show()


# In[247]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif=pd.DataFrame()
vif['vif']=[variance_inflation_factor(x_scaled,i) for i in range(x_scaled.shape[1])]
vif['features']=x.columns

vif
# greater than 5 means high multi-collinearity


# In[250]:


from sklearn.decomposition import PCA
pca=PCA()
principal_components=pca.fit_transform(x_scaled)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('no of components')
plt.ylabel('% covariance')#for each component
plt.title('Covariance explained')
plt.show()

12 components can completely describe data. 95% can be explained by 10 components.
# In[253]:


pca=PCA(n_components=10)
new_data=pca.fit_transform(x_scaled)

principla_x=pd.DataFrame(new_data,columns=['1','2','3','4','5','6','7','8','9','10'])


# In[254]:


principla_x


# In[255]:


p_train, p_test, py_train, py_test = train_test_split( principla_x, y, test_size=0.3)

lm2.fit(p_train, py_train)


# In[256]:


p_pred=lm2.predict(p_test)
r2_score(py_test,p_pred)


# In[267]:



#import required packages
from sklearn import neighbors
from math import sqrt
get_ipython().run_line_magic('matplotlib', 'inline')
rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, y_train)  #fit the model
    pred=model.predict(X_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)


# In[268]:


#plotting the rmse values against k values
curve = pd.DataFrame(rmse_val) #elbow curve 
curve.plot()

ERROR minimum at k=3
# In[271]:


model = neighbors.KNeighborsRegressor(n_neighbors = 3)

model.fit(X_train, y_train)  #fit the model
pred=model.predict(X_test) #make prediction on test set


# In[272]:


model.score(X_test,y_test)


# In[274]:


from sklearn.tree import DecisionTreeRegressor

# Fit regression model
regr = DecisionTreeRegressor(max_depth=2)
regr.fit(X_train, y_train)


# In[276]:


result = regr.predict(X_test)
regr.score(X_test,y_test)


# In[ ]:




