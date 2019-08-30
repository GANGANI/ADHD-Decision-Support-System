from med2image import med2image
from keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np
import os

'''
c_convert=med2image.med2image_nii(inputFile="falff_0010014_session_1_rest_1.nii.gz", outputDir="temp9",outputFileStem="image",outputFileType="png", sliceToConvert='-1',frameToConvert='0',showSlices=False, reslice=False)
med2image.misc.tic()
c_convert.run()
'''

model=load_model('first_try_model.h5')
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

images=[]

for img in os.listdir('/home/adhd/adhd_cnn/dataFolder/temp9/'):
    img=cv2.imread('temp9/'+img)
    #img=img.astype('float')/255.0
    img=cv2.resize(img,(73,61))
    img=np.reshape(img,[1,73,61,3])
    images.append(img)

images=np.vstack(images)

cls=model.predict_classes(images,batch_size=10)

print(cls)
print('Possibility of ADHD: ', (cls==0).sum()/len(cls))
print('Possibility of non-ADHD: ', (cls==1).sum()/len(cls))
'''
img=cv2.imread('temp2/image-frame000-slice000.png')
#img=img.astype("float")/255.0
img=cv2.resize(img,(73,61))
img=np.reshape(img,[1,73,61,3])
cls=model.predict_classes(img)
print (cls)
'''

