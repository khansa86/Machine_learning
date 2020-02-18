# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:22:36 2020

@author: hinaa
"""
from plt import Image
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
classifier=load_model('expressions1.h5')
test_image = image.load_img('D:\ssmile.png', target_size = (100, 100))
test_image = image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result = classifier.predict(test_image)
Image.open('D:\ssmile.png')
name_of_classes = [ "angry", "happy", "sad"]
no_of_classes=len(name_of_classes)
for i in range(no_of_classes):
    if (result[0][i] == 1.0):
        print ('Predicted:',name_of_classes[i])
    