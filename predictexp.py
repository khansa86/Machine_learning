# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 12:08:46 2020

@author: hinaa
"""
from keras.preprocessing import image
from matplotlib import pyplot as plt
from keras.models import load_model
classifier=load_model('D:\expressions1.h5')
import numpy as np

def emotion_analysis(emotions):
    objects = ('angry', 'happy', 'sad')
    y_pos = np.arange(len(objects))
    print('prediction= '+(max(objects)))
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')

    plt.show()
     
  
img = image.load_img("D:\sad.png", target_size=(48, 48))
 
test_image = image.img_to_array(img)
test_image = np.expand_dims(test_image, axis = 0)
 
test_image /= 255
 
custom =classifier.predict(test_image)
emotion_analysis(custom[0])
 

plt.show()

