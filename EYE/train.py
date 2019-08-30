import pandas as pd

#read in training data
train_df_2 = pd.read_csv('/home/adhd/EYE/dataset/dataset.csv')

#view data structure
train_df_2.head()

#create a dataframe with all training data except the target column
train_X_2 = train_df_2.drop(columns=['Class'])

#check that the target variable has been removed
train_X_2.head()

from keras.utils import to_categorical
#one-hot encode target column
train_y_2 = to_categorical(train_df_2.Class, dtype = 'float32')

#vcheck that target column has been converted
train_y_2[0:12469]

#create model
model_2 = Sequential()

#get number of columns in training data
n_cols_2 = train_X_2.shape[1]

#add layers to model
model_2.add(Dense(14, activation='relu', input_shape=(n_cols_2,)))
model_2.add(Dense(14, activation='relu'))
model_2.add(Dense(14, activation='relu'))
model_2.add(Dense(1, activation='softmax'))

#compile model using accuracy to measure model performance
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train model
model_2.fit(X_2, target, epochs=30, validation_split=0.2, callbacks=[early_stopping_monitor])
