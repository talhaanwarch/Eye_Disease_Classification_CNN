# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:17:37 2019

@author: Talha
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

eye_cascade = cv2.CascadeClassifier('D:/Datasets/Image dataset/eye disease/haarcascade_eye.xml')

#read directory
import glob
#dirc=glob.glob('D:/Datasets/Image dataset/eye disease/eye-disease-dataset/Eye_diseases/*/')

path='D:/Datasets/Image dataset/eye disease/eye-disease-dataset/'
os.mkdir(path+'Cropped_Images/')
dir_names=os.listdir(path+'Eye_diseases/')
for dir_name in dir_names:
    counter=0
    os.mkdir(path+'Cropped_Images/'+dir_name)
    img_path=glob.glob(path+'Eye_diseases/{}/*.jpeg'.format(dir_name))
    for img in img_path:
        image=cv2.imread(img)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        eyes = eye_cascade.detectMultiScale(image)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(image,(ex,ey),(ex+ew,ey+eh),(255,255,255),1)
            #plt.imshow(image)

            crp=image[ey:ey+eh,ex:ex+ew,:]
            #plt.imshow(crp)
            plt.imsave(path+'Cropped_Images/{}/image_{}.jpg'.format(dir_name,counter),crp)
            counter+=1
    









