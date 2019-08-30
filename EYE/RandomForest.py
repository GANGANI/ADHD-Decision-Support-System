import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv('dataset2.csv')

dataset.head()

dataset.Class[dataset.Class=='ADHD']=1
dataset.Class[dataset.Class=='Non-ADHD']=0

dataset['Gender']=pd.Categorical(dataset['Gender'])
datasetDummies=pd.get_dummies(dataset['Gender'],prefix='Gender')
dataset=dataset.drop('Gender',axis=1)
dataset=pd.concat([dataset,datasetDummies],axis=1)
print(dataset.head())

Y=dataset.Class
Y=Y.astype('int')
X= dataset.drop('Class',axis=1)

from sklearn.model_selection import train_test_split
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.25, random_state=0)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=1000, random_state=42)   
rf.fit(X_train, Y_train)  
y_pred = rf.predict(X_test)  

# Calculate the absolute errors
errors = abs(y_pred - Y_test) 

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / Y_test)

print (mape)
print (Y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
