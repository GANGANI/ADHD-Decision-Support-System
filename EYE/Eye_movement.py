# organize imports
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# seed for reproducing same results
seed = 9
np.random.seed(seed)

# load pima indians dataset
#dataset = np.loadtxt('/home/adhd/EYE/dataset/dataset.csv', dtype={'names': ('Gender','NumberOfFixations','FixationDuration(ms)','FixationDurationAvg(ms)','FixationDurationStd(ms)','NumberOfSaccades','SaccadeDuration(ms)','SaccadeDurationAvg(ms)','SaccadesDurationStd(ms)','GazePointX(ADCSpx)','GazePointY(ADCSpx)','PupilDiameterLeft(mm)','PupilDiameterRight(mm)','Class'), 'formats': ('S1', 'int', 'int', 'float', 'float', 'int', 'int', 'float', 'float', 'float', 'float', 'float', 'float', 'S1' )}, delimiter=',', skiprows=1)
print('exo')
dataset =pd.read_csv('dataset.csv')
# split into input and output variables
'''X = dataset[:,:1]
Y = dataset[:,14]'''

dataset.Class[dataset.Class=='ADHD']=1
dataset.Class[dataset.Class=='Non-ADHD']=0

dataset['Gender']=pd.Categorical(dataset['Gender'])
datasetDummies=pd.get_dummies(dataset['Gender'],prefix='Gender')
dataset=dataset.drop('Gender',axis=1)
dataset=pd.concat([dataset,datasetDummies],axis=1)
print(dataset.head())

Y=dataset.Class
X= dataset.drop('Class',axis=1)

print (list(dataset))
print (dataset.shape)

#split the data into training (67%) and testing (33%)
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.33, random_state=seed)

# create the model
model = Sequential()
model.add(Dense(15, input_dim=15, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history=model.fit(X_train, Y_train,
          batch_size=100,
          epochs=120,
          verbose=1,
          validation_data=(X_test, Y_test))

# evaluate the model
scores = model.evaluate(X_test, Y_test)
print ("Accuracy: %.2f%%" %(scores[1]*100))

model.save('eye_first_try_model.h5')