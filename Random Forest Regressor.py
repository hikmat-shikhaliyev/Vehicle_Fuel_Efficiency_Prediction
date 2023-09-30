#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# In[2]:


data=pd.read_csv(r'C:\Users\ASUS\Downloads\auto-mpg.csv')
data


# In[3]:


data.describe(include='all')


# In[4]:


data=data.drop('car name', axis=1)


# In[5]:


data.isnull().sum()


# In[6]:


data.corr()['mpg']


# In[7]:


data.dtypes


# In[13]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables=data[[
#                 'cylinders', 
                'displacement', 
#                 'horsepower', 
#                 'weight', 
#                 'acceleration', 
#                 'model year', 
                'origin']]

vif=pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["Features"] = variables.columns
vif


# In[14]:


data=data.drop('cylinders', axis=1)
data=data.drop('horsepower', axis=1)
data=data.drop('weight', axis=1)
data=data.drop('acceleration', axis=1)
data=data.drop('model year', axis=1)


# In[15]:


data.head()


# In[16]:


for i in data[['displacement', 'origin', 'mpg']]:
    sns.boxplot(x=data[i], data=data)
    plt.show()


# In[17]:


q1=data.quantile(0.25)
q3=data.quantile(0.75)
IQR=q3-q1
Lower=q1-1.5*IQR
Upper=q3+1.5*IQR


# In[18]:


for i in data[['displacement', 'origin', 'mpg']]:
    data[i] = np.where(data[i] > Upper[i], Upper[i],data[i])
    data[i] = np.where(data[i] < Lower[i], Lower[i],data[i]) 


# In[19]:


for i in data[['displacement', 'origin', 'mpg']]:
    sns.boxplot(x=data[i], data=data)
    plt.show()


# In[20]:


data=data.reset_index(drop=True)


# In[21]:


data.head()


# In[24]:


X=data.drop('mpg', axis=1)
y=data['mpg']


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[31]:


from sklearn import metrics
def evaluate(model, X_test, y_test):
    y_pred=model.predict(X_test)
    mae_test=metrics.mean_absolute_error(y_test, y_pred)
    mse_test=metrics.mean_squared_error(y_test, y_pred)
    rmse_test=np.sqrt(mse_test)
    r2_test=metrics.r2_score(y_test, y_pred)
    
    y_pred_train=model.predict(X_train)
    mae_train=metrics.mean_absolute_error(y_train, y_pred_train)
    mse_train=metrics.mean_squared_error(y_train, y_pred_train)
    rmse_train=np.sqrt(mse_train)
    r2_train=metrics.r2_score(y_train, y_pred_train)
    
    
    results_dict = {
        'Metric': ['MAE', 'MSE', 'RMSE', 'R2'],
        'Train': [mae_train, mse_train, rmse_train, r2_train*100],
        'Test': [mae_test, mse_test, rmse_test, r2_test*100]
    }

    results_df = pd.DataFrame(results_dict)
    
    print(results_df)


# In[27]:


rfr=RandomForestRegressor(n_estimators=100, random_state=42)
rfr.fit(X_train, y_train)


# In[32]:


result=evaluate(rfr, X_test, y_test)


# In[33]:


from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[34]:


rfr_randomized = RandomizedSearchCV(estimator = rfr, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=1, random_state=42, n_jobs = -1)

rfr_randomized.fit(X_train, y_train)


# In[35]:


rfr_randomized.best_params_


# In[37]:


optimized_model=rfr_randomized.best_estimator_
optimized_accuracy=evaluate(optimized_model, X_test, y_test)


# In[38]:


variables = []
train_r2_scores = []
test_r2_scores = []

for i in X_train.columns: 
    X_train_single = X_train[[i]]
    X_test_single = X_test[[i]]

    
    optimized_model.fit(X_train_single, y_train)
    
    
    y_pred_train_single = optimized_model.predict(X_train_single)
    train_r2 = metrics.r2_score(y_train, y_pred_train_single)
    
    

    y_pred_test_single = optimized_model.predict(X_test_single)
    test_r2 = metrics.r2_score(y_test, y_pred_test_single)

    variables.append(i)
    train_r2_scores.append(train_r2)
    test_r2_scores.append(test_r2)
    
    
    
results_df = pd.DataFrame({'Variable': variables, 'Train R2': train_r2_scores, 'Test R2': test_r2_scores})

results_df_sorted = results_df.sort_values(by='Test R2', ascending=False)

pd.options.display.float_format = '{:.4f}'.format

results_df_sorted


# In[ ]:




