# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:49:12 2018

@author: Administrator
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

#Open Camera object
cap = cv.VideoCapture(0)

#Decrease frame size
cap.set(cv.CAP_PROP_FRAME_WIDTH, 700)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 200)

i=0
while(1):
    if(cv.waitKey(2)==99):
        i=i+1
        ret, frame = cap.read()

        cv.rectangle(frame, (500,500), (100,100), (0,255,0),0)
        crop_frame=frame[100:500,100:500]
         #Blur the image
        #blur = cv2.blur(crop_frame,(3,3))c
#        blur = cv.GaussianBlur(crop_frame, (3,3), 0)    
#            #Convert to HSV color space
#        hsv = cv.cvtColor(blur,cv.COLOR_BGR2HSV)
#        
#        #Create a binary image with where white will be skin colors and rest is black
#        mask2 = cv.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))
#        #cv2.imshow('Masked',mask2)
#
#        kernel_square = np.ones((11,11),np.uint8)
#        kernel_ellipse= cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
#        med=cv.medianBlur(mask2,5)
            
#        # RGB到YCbCr色彩空间
#        image_YCbCr = cv.cvtColor(crop_frame, cv.COLOR_RGB2YCrCb)
#        
#        # 返回行数，列数，通道个数
#        shape = image_YCbCr.shape
#        
#        Kl, Kh = 125, 188
#        Ymin, Ymax = 16, 235
#        Wlcb, Wlcr = 23, 20
#        Whcb, Whcr = 14, 10
#        Wcb, Wcr = 46.97, 38.76
#        # 椭圆模型参数
#        Cx, Cy = 109.38, 152.02
#        ecx, ecy = 1.60, 2.41
#        a, b = 25.39, 14.03
#        Theta = 2.53 / np.pi * 180
#        # 每行
#        for row in range(shape[0]):
#            # 每列
#            for col in range(shape[1]):
#                Y = image_YCbCr[row, col, 0]
#                CbY = image_YCbCr[row, col, 1]
#                CrY = image_YCbCr[row, col, 2]
#                if Y < Kl or Y > Kh:
#                    # 求Cb, Cr的均值
#                    if Y < Kl:
#                        # 公式(7)
#                        CbY_aver = 108 + (Kl - Y) * (118 - 108) / (Kl - Ymin)
#                        # 公式(8)
#                        CrY_aver = 154 - (Kl - Y) * (154 - 144) / (Kl - Ymin)
#                        # 公式(6)
#                        WcbY = Wlcb + (Y - Ymin) * (Wcb - Wlcb) / (Kl - Ymin)
#                        WcrY = Wlcr + (Y - Ymin) * (Wcr - Wlcr) / (Kl - Ymin)
#                    elif Y > Kh:
#                        # 公式(7)
#                        CbY_aver = 108 + (Y - Kh) * (118 - 108) / (Ymax - Kh)
#                        # 公式(8)
#                        CrY_aver = 154 + (Y - Kh) * (154 - 132) / (Ymax - Kh)
#                        # 公式(6)
#                        WcbY = Whcb + (Ymax - Y) * (Wcb - Whcb) / (Ymax - Kh)
#                        WcrY = Whcr + (Ymax - Y) * (Wcr - Whcr) / (Ymax - Kh)
#                    # 求Cb(Kh), Cr(Kh)的均值
#                    CbKh_aver = 108 + (Kh - Kh) * (118 - 108) / (Ymax - Kh)
#                    CrKh_aver = 154 + (Kh - Kh) * (154 - 132) / (Ymax - Kh)
#                    # 公式(5)
#                    Cb = (CbY - CbY_aver) * Wcb / WcbY + CbKh_aver
#                    Cr = (CrY - CrY_aver) * Wcr / WcrY + CrKh_aver
#                else:
#                    # 公式(5)
#                    Cb = CbY
#                    Cr = CrY
#                # Cb，Cr代入椭圆模型
#                cosTheta = np.cos(Theta)
#                sinTehta = np.sin(Theta)
#                matrixA = np.array([[cosTheta, sinTehta], [-sinTehta, cosTheta]], dtype=np.double)
#                matrixB = np.array([[Cb - Cx], [Cr - Cy]], dtype=np.double)
#                # 矩阵相乘
#                matrixC = np.dot(matrixA, matrixB)
#                x = matrixC[0, 0]
#                y = matrixC[1, 0]
#                ellipse = (x - ecx) ** 2 / a ** 2 + (y - ecy) ** 2 / b ** 2
#                if ellipse <= 1:
#                    # 白
#                    image_YCbCr[row, col] = [255, 255, 255]
#                    # 黑
#                else:
#                    image_YCbCr[row, col] = [0, 0, 0]
#            
#        cv.imshow('main',frame)
#        cv.imshow('masked',image_YCbCr)
#    
#        
#    #cv2.imshow('masked_',thresh)
#        res_size=cv.resize(image_YCbCr,(80,80))
        rows,cols,channels = crop_frame.shape  
          
        # prepare an empty image space  
        imgSkin = np.zeros(crop_frame.shape, np.uint8)  
        # copy original image  
        imgSkin = crop_frame.copy()  
          
        for r in range(rows):  
            for c in range(cols):  
           
                # get pixel value         
                B = crop_frame.item(r,c,0)  
                G = crop_frame.item(r,c,1)  
                R = crop_frame.item(r,c,2)  
                  
                # non-skin area if skin equals 0, skin area otherwise          
                skin = 0  
                          
                if (abs(R - G) > 15) and (R > G) and (R > B):  
                    if (R > 95) and (G > 40) and (B > 20) and (max(R,G,B) - min(R,G,B) > 15):                 
                        skin = 1      
                        # print 'Condition 1 satisfied!'  
                    elif (R > 220) and (G > 210) and (B > 170):  
                        skin = 1  
                        # print 'Condition 2 satisfied!'  
                  
                if 0 == skin:  
                    imgSkin.itemset((r,c,0),0)  
                    imgSkin.itemset((r,c,1),0)                  
                    imgSkin.itemset((r,c,2),0)  
                    # print 'Skin detected!'  
          
        # convert color space of images because of the display difference between cv2 and matplotlib                           
        crop_frame = cv.cvtColor(crop_frame, cv.COLOR_BGR2RGB)  
        imgSkin = cv.cvtColor(imgSkin, cv.COLOR_BGR2RGB)
        cv.imshow('main',frame)
        cv.imshow('masked',imgSkin)
        
        (r,g,b)=cv.split(imgSkin)
        imgSkin=cv.merge([b,g,r])
        res_size=cv.resize(imgSkin,(70,50))
        cv.imwrite(os.path.join('./background',"gest"+str(i)+".jpg"),res_size)
        #close the output video by pressing 'ESC'
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break
    else:
        ret, frame = cap.read()
        
        cv.rectangle(frame, (500,500), (100,100), (0,255,0),0)
        crop_frame=frame[100:500,100:500]
         #Blur the image
        blur = cv.blur(crop_frame,(3,3))
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
    
#        rows,cols,channels = crop_frame.shape  
#          
#        # prepare an empty image space  
#        imgSkin = np.zeros(crop_frame.shape, np.uint8)  
#        # copy original image  
#        imgSkin = crop_frame.copy()  
#          
#        for r in range(rows):  
#            for c in range(cols):  
#           
#                # get pixel value         
#                B = crop_frame.item(r,c,0)  
#                G = crop_frame.item(r,c,1)  
#                R = crop_frame.item(r,c,2)  
#                  
#                # non-skin area if skin equals 0, skin area otherwise          
#                skin = 0  
#                          
#                if (abs(R - G) > 15) and (R > G) and (R > B):  
#                    if (R > 95) and (G > 40) and (B > 20) and (max(R,G,B) - min(R,G,B) > 15):                 
#                        skin = 1      
#                        # print 'Condition 1 satisfied!'  
#                    elif (R > 220) and (G > 210) and (B > 170):  
#                        skin = 1  
#                        # print 'Condition 2 satisfied!'  
#                  
#                if 0 == skin:  
#                    imgSkin.itemset((r,c,0),0)  
#                    imgSkin.itemset((r,c,1),0)                  
#                    imgSkin.itemset((r,c,2),0)  
#                    # print 'Skin detected!'  
#          
#        # convert color space of images because of the display difference between cv2 and matplotlib                           
#        crop_frame = cv.cvtColor(crop_frame, cv.COLOR_BGR2RGB)  
#        imgSkin = cv.cvtColor(imgSkin, cv.COLOR_BGR2RGB)
#        cv.imshow('main',frame)
#        cv.imshow('masked',imgSkin)
     #close the output video by pressing 'ESC'
        k = cv.waitKey(2) & 0xFF
        if k == 27:
            break
cap.release()
cv.destroyAllWindows()
