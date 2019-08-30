import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import tree 
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn import metrics

import pickle
from sklearn.externals import joblib

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

(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.33, random_state=0)

model = tree.DecisionTreeClassifier() 
model = model.fit(X_train, Y_train) 
y_pred = model.predict(X_test) 

print(confusion_matrix(Y_test,y_pred))  
print(classification_report(Y_test,y_pred))
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))

joblib.dump(model, 'model.pkl') 

#Accuracy: 0.8558930741190766
#Accuracy: 0.5667071688942892



