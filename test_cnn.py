import os
import cv2
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,Conv3D,MaxPooling2D,MaxPooling3D,UpSampling2D,Conv2DTranspose,Conv3DTranspose,UpSampling3D
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
import numpy as np
import scipy.misc
import numpy.random as rng
from PIL import Image, ImageDraw, ImageFont
from sklearn.utils import shuffle
import nibabel as nib #reading MR images
from sklearn.model_selection import train_test_split
#import mathcr
import glob
from matplotlib import pyplot as plt
#%matplotlib inline
import tensorflow as tf

ff = glob.glob('dataFolder/*')
print(ff[0])
print (len(ff))
ini_labels=[3,0,0,2,1]

images = []
labels=[]
for f in range(len(ff)):
    a = nib.load(ff[f])
    a = a.get_data()
    #a = a[:,:,:,75:126]
    for i in range(a.shape[3]):
        images.append((a[:,:,:,i]))
        labels.append(ini_labels[f])
print (a.shape)

images = np.asarray(images)
labels= np.asarray(labels)
print(images.shape)
print(labels.shape)

train_x,test_x,train_y,test_y = train_test_split(images,labels,test_size=0.2,random_state=13)
print("Dataset (images) shape: {shape}".format(shape=images.shape))
print ("Train data : {shape} ".format(shape=train_x.shape))


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  print("features_x: {shape}".format(shape=features["x"]))
  input_layer = tf.reshape(features["x"], [-1, 64, 64,39, 1])
  print("input_layer : {shape}".format(shape=input_layer.shape))

  # Convolutional Layer #1
  conv1 = tf.layers.conv3d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5,5],
      padding="same",
      activation=tf.nn.relu)
  print("conv1 : {shape}".format(shape=conv1.shape))

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[2, 2,2], strides=2)
  print("pool1 : {shape}".format(shape=pool1.shape))

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv3d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5,5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2,2], strides=2)
  print("conv2 : {shape}".format(shape=conv2.shape))
  print("pool2 : {shape}".format(shape=pool2.shape))

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1,  16* 16 *9* 64])
  print("pool2_flat : {shape}".format(shape=pool2_flat.shape))
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)
  print("logits : {shape}".format(shape=logits.shape))

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  print("Labels {shape}".format(shape=labels.shape))
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

#estimator
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_x.astype(np.float32)},
    y=train_y,
    batch_size=2,
    num_epochs=1,
    shuffle=True)


# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}

logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

# train one step and display the probabilties
# mnist_classifier.train(input_fn=train_input_fn,steps=1,hooks=[logging_hook])

mnist_classifier.train(input_fn=train_input_fn, steps=100)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": test_x.astype(np.float32)},
    y=test_y,
    num_epochs=1,
    shuffle=False)

eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)
