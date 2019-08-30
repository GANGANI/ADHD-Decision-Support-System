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

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,Y_train)
predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(Y_test,predictions))

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(Y_test, predictions))

