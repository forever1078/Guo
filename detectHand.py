# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# -*- coding: cp936 -*-
import cv2
filename='facehand.jpg'
def detect (filename):
    hand_cascade=cv2.CascadeClassifier('xml.xml')
    img=cv2.imread(filename)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hands=hand_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in hands:
        img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        break#这个地方加上break是考虑到应该只有一个手势就不要继续匹配下去了
    cv2.imwrite('hand.jpg',img)

detect(filename)
