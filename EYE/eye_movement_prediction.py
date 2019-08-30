from med2image import med2image
from keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np
import os
from flask import Flask, render_template, request, flash, redirect, request, jsonify
import tensorflow as tf
import pandas as pd

model=load_model('eye_first_try_model.h5')
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

known_categories = ['Saccade','Fixation','Unclassified']
dataset =pd.read_csv('test.csv')
# split into input and output variables
'''X = dataset[:,:1]
Y = dataset[:,14]'''

dataset['Gender']=pd.Categorical(dataset['Gender'], known_categories)
datasetDummies=pd.get_dummies(dataset['Gender'],prefix='Gender')
dataset=dataset.drop('Gender',axis=1)
dataset=pd.concat([dataset,datasetDummies],axis=1)
print(dataset.shape)
#dataset['temp']=0
print(dataset.head())

print(dataset.shape)
cls=model.predict_classes(dataset)
print(cls)

