# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:23:38 2018

@author: Administrator
"""

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import numpy as np
import os
import cv2 as cv
best_model = load_model('1.weights-improvement-02-0.98.h5')

#Open Camera object
cap = cv.VideoCapture(0)

#Decrease frame size
cap.set(cv.CAP_PROP_FRAME_WIDTH, 700)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 400)

i=0
while(1):
    if(cv.waitKey(2)==99):
        i=i+1
        ret, frame = cap.read()

        cv.rectangle(frame, (300,300), (100,100), (0,255,0),0)
        crop_frame=frame[100:300,100:300]
         #Blur the image
        #blur = cv2.blur(crop_frame,(3,3))c
        blur = cv.GaussianBlur(crop_frame, (3,3), 0)    
            #Convert to HSV color space
        hsv = cv.cvtColor(blur,cv.COLOR_BGR2HSV)
        
        #Create a binary image with where white will be skin colors and rest is black
        mask2 = cv.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))
        #cv2.imshow('Masked',mask2)

        kernel_square = np.ones((11,11),np.uint8)
        kernel_ellipse= cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
        med=cv.medianBlur(mask2,5)
            
            
        cv.imshow('main',frame)
        cv.imshow('masked',med)
    
        
    #cv2.imshow('masked_',thresh)
        res_size=cv.resize(med,(50,50))
        cv.imwrite(os.path.join('./',"gest"+str(i)+".jpg"),res_size)
        cv.imshow('res',res_size)
        image_path = './'+"gest"+str(i)+".jpg"
        img = image.load_img(image_path,target_size=(224,224))
        
        x = image.img_to_array(img)
        ##x.resize(100,100,3)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds=best_model.predict(x)
        print ('Predicted:',preds)
        sStr2 = ",".join(map(str, preds[0]))
        sq=sStr2.split(",")
        for i in range(len(sq)):
            sq[i]=float(sq[i])
        maxdata=sq[0]
        maxsequence=0
        for j in range(len(sq)):
            if sq[j]>maxdata:
                maxdata=sq[j]
                maxsequence=j
        print ('this gesture belongs to gesture :', maxsequence)
#        #preds = best_model.predict(x,1,2)
#        preds=best_model.predict_classes(x)
#        print ('Predicted:', preds)
        
     #close the output video by pressing 'ESC'
        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break
    else:
        ret, frame = cap.read()
        
        cv.rectangle(frame, (300,300), (100,100), (0,255,0),0)
        crop_frame=frame[100:300,100:300]
         #Blur the image
        #blur = cv2.blur(crop_frame,(3,3))
        blur = cv.GaussianBlur(crop_frame, (3, 3), 0)    
            #Convert to HSV color space
        hsv = cv.cvtColor(blur,cv.COLOR_BGR2HSV)
         
        #Create a binary image with where white will be skin colors and rest is black
        mask2 = cv.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))
        #cv2.imshow('Masked',mask2)

        kernel_square = np.ones((11,11),np.uint8)
        kernel_ellipse= cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
        med=cv.medianBlur(mask2,5)

        
        cv.imshow('main',frame)
        cv.imshow('masked',med)
    
        
     #close the output video by pressing 'ESC'
        k = cv.waitKey(2) & 0xFF
        if k == 27:
            break
cap.release()
cv.destroyAllWindows()


