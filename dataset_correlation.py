import tensorflow as tf
import scipy.ndimage
from scipy.misc import imsave
import matplotlib.pyplot as plt
import numpy as np
import glob
import nibabel as nib #reading MR images
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import pandas as pd
import seaborn as sns

plt.switch_backend('agg')

ff = glob.glob('dataFolder/*')
ini_labels=[3,3,0,0,0,2,2,2,0,3,3,0,0,0,0,0,0]
#ini_labels=['inattentive','inattentive','normal','normal','normal','hyperactivity','hyperactivity']
batch_size = 4
num_classes = 4
epochs = 100

images = []
labels=[]
for f in range(len(ff)):
    a = nib.load(ff[f])
    a = a.get_data()
    print('a images shape: {shape}'.format(shape=a.shape))
    images.append(a)

images = np.asarray(images)
images=images.reshape(5*64*64*39,192)
imgs=pd.DataFrame(data=images[1:,1:],index=images[1:,0],columns=images[0,1:])
#images=images.reshape(5,64,47,1)
labels = np.asarray(ini_labels)
print('images shape: {shape}'.format(shape=images.shape))
print('labels shape: {shape}'.format(shape=labels.shape))
svm=sns.heatmap(imgs.corr(), annot=True)

figure = svm.get_figure()    
figure.savefig('svm_conf.png', dpi=400)
