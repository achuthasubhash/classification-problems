# -*- coding: utf-8 -*-
"""
Created on Thu May 14 22:06:30 2020

@author: DELL
"""
from keras.models import load_model
classifier = load_model("Classifier.h5")

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('E:\\github\\my upload\\covid\\classify mask\\observations-master\\experiements\\dest_folder\\test\\with_mask\\5-with-mask.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image) #2D to 3D (to I/P shape)
test_image = np.expand_dims(test_image, axis = 0) #axis specify the poistion of index where add  new DIM &expand add 1 New Dim
result = classifier.predict(test_image)
if result[0][0] == 0:
    prediction = 'nomask'
else:
    prediction = 'mask'
