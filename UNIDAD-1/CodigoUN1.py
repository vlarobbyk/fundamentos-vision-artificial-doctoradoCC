import cv2
import os

import matplotlib.pyplot as pp
import numpy as np


class CodigoUN1:

    def __init__(self):
        print('Test')

    
    def read_image(self):
        img = cv2.imread('San-Basilio.jpg')
        print(img.shape)

        pp.imshow(img)
        pp.show()

        #cv2.imshow('Imagen', img)
        #cv2.waitKey(0)

    def create_img(self):
        #img = np.full((4,4), 255, dtype = np.uint8)
        img = np.ones((4,4), dtype=np.uint8)*255
        
        print(img.shape)

        print(img)
        
        pp.imshow(img, cmap='gray', vmin=0, vmax=255)
        pp.axis('off')
        pp.show()


    def motion_detector(self):
        video = cv2.VideoCapture(0)
        if video.isOpened():
            print('Open ... ')

            frame = None
            frame_ant = None
            resta = None
            primero = True

            while(3==3):
                ret, frame = video.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if primero==True:
                    primero = False
                    frame_ant = np.copy(frame)

                resta = cv2.absdiff(frame_ant,frame)
                frame_ant = np.copy(frame)
                cv2.imshow('Video', resta)

                if cv2.waitKey(23)==27:
                    break
            
            video.release()
            cv2.destroyAllWindows()


            return
        
        print('Can\'t open video')


if __name__=="__main__":
    cod = CodigoUN1()
    #cod.read_image()
    #cod.create_img()
    cod.motion_detector()