
# coding: utf-8

# In[6]:


import pandas as pd
from sklearn import svm
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics

X=pd.read_csv('dengue_features_train.csv')
y=pd.read_csv('dengue_labels_train.csv')
x_test=pd.read_csv('dengue_features_test.csv')

y=y['total_cases']

X=X.fillna(X.mean())

X=X.drop('week_start_date',1)
one_hot=pd.get_dummies(X['city'])
X=X.drop('city',1)
X=X.join(one_hot)
X=X.drop('reanalysis_max_air_temp_k',1)
X=X.drop('reanalysis_min_air_temp_k',1)
X=X.drop('station_max_temp_c',1)
X=X.drop('station_min_temp_c',1)
X=X.drop('precipitation_amt_mm',1)
X=X.drop('reanalysis_precip_amt_kg_per_m2',1)
X=X.drop('reanalysis_sat_precip_amt_mm',1)
X=X.drop('station_precip_mm',1)
X=X.drop('ndvi_sw',1)
X=X.drop('station_avg_temp_c',1)

print(list(X))
#normalize ndvi_ne
X['ndvi_ne']=(X['ndvi_ne']-X['ndvi_ne'].mean())/(X['ndvi_ne'].max()-X['ndvi_ne'].min())
#normalize ndvi_nw
#X['ndvi_nw']=(X['ndvi_nw']-X['ndvi_nw'].mean())/(X['ndvi_nw'].max()-X['ndvi_nw'].min())
X['reanalysis_relative_humidity_percent']=(X['reanalysis_relative_humidity_percent']-X['reanalysis_relative_humidity_percent'].mean())/(X['reanalysis_relative_humidity_percent'].max()-X['reanalysis_relative_humidity_percent'].min())

x_test=x_test.fillna(x_test.mean())
x_test=x_test.drop('week_start_date',1)
one_hot=pd.get_dummies(x_test['city'])
x_test=x_test.drop('city',1)
x_test=x_test.join(one_hot)
x_test=x_test.drop('reanalysis_max_air_temp_k',1)
x_test=x_test.drop('reanalysis_min_air_temp_k',1)
x_test=x_test.drop('station_max_temp_c',1)
x_test=x_test.drop('station_min_temp_c',1)
x_test=x_test.drop('precipitation_amt_mm',1)
x_test=x_test.drop('reanalysis_precip_amt_kg_per_m2',1)
x_test=x_test.drop('reanalysis_sat_precip_amt_mm',1)
x_test=x_test.drop('station_precip_mm',1)
x_test=x_test.drop('ndvi_sw',1)
x_test=x_test.drop('station_avg_temp_c',1)
# x_test['ndvi_ne']=(x_test['ndvi_ne']-x_test['ndvi_ne'].mean())/(x_test['ndvi_ne'].max()-x_test['ndvi_ne'].min())
# x_test['ndvi_nw']=(x_test['ndvi_nw']-x_test['ndvi_nw'].mean())/(x_test['ndvi_nw'].max()-x_test['ndvi_nw'].min())
# print(X.shape)
# print(y.shape)

#x_test['reanalysis_relative_humidity_percent']=(x_test['reanalysis_relative_humidity_percent']-x_test['reanalysis_relative_humidity_percent'].mean())/(x_test['reanalysis_relative_humidity_percent'].max()-x_test['reanalysis_relative_humidity_percent'].min())

# X['total_cases']=y
# print(X.corr())

plt.matshow(X.corr())
plt.show()

print(X.shape)
print(y.shape)

#linear kernel-> 35
model=tree.DecisionTreeClassifier() 
#rbf kernel-> not working
#model=svm.SVC(kernel='rbf',C=1, gamma=0.7)
model.fit(X,y)
model.score(X,y)
predicted=model.predict(x_test)

y_test=pd.read_csv('submission_format.csv')
y_test['total_cases']=predicted
y_test.to_csv('tenth_try.csv',index=False)

kfold=KFold(n_splits=3,random_state=7)
rst=cross_val_score(model,X,y,cv=kfold,scoring='accuracy')
print(rst.mean())

print("Accuracy:",metrics.accuracy_score(y_test, predicted))

# In[6]:


y_test=pd.read_csv('submission_format.csv')
y_test['total_cases']=predicted
y_test.to_csv('thirtd_try.csv',index=False)

