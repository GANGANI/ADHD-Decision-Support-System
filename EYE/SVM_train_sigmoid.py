import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv('dataset.csv')

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
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.33, random_state=0)

from sklearn.svm import SVC  
svclassifier = SVC(kernel='sigmoid')  
svclassifier.fit(X_train, Y_train)  

y_pred = svclassifier.predict(X_test)  

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(Y_test,y_pred))  
print(classification_report(Y_test,y_pred))

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))

#Accuracy: 0.5064398541919806



