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


    def evento_track(self, valor):
        pass

    
    def seleccion_pixeles_rango(self):
        cv2.namedWindow('Original', cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar('HMin','Original', 0, 180, self.evento_track)
        cv2.createTrackbar('HMax','Original', 0, 180, self.evento_track)

        cv2.createTrackbar('SMin','Original', 0, 255, self.evento_track)
        cv2.createTrackbar('SMax','Original', 0, 255, self.evento_track)

        cv2.createTrackbar('VMin','Original', 0, 255, self.evento_track)
        cv2.createTrackbar('VMax','Original', 0, 255, self.evento_track)

        frame = None
        video = cv2.VideoCapture(0)

        frame_hsv = None
        frame_binario = None

        mascara = None
        vacio = True

        if video.isOpened():
            while 3==3:
                ret, frame = video.read()
                frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                if vacio:
                    vacio = False
                    mascara = np.full((frame.shape[0], frame.shape[1], 1), 0, dtype=np.uint8)
                    cv2.ellipse(mascara, 
                                (frame.shape[0]//2, frame.shape[1]//2), 
                                (100,100), 20, 0, 360, (255), -1)

                frame_binario = cv2.inRange(frame_hsv, 
                    (
                        cv2.getTrackbarPos('HMin', 'Original'),
                        cv2.getTrackbarPos('SMin', 'Original'),
                        cv2.getTrackbarPos('VMin', 'Original'),
                    ),
                    (
                        cv2.getTrackbarPos('HMax', 'Original'),
                        cv2.getTrackbarPos('SMax', 'Original'),
                        cv2.getTrackbarPos('VMax', 'Original'),
                    )
                )

                frame_and = cv2.bitwise_and(frame, frame, mask=frame_binario)

                cv2.imshow('Original', frame)
                cv2.imshow('HSV', frame_hsv)
                cv2.imshow('Binario', frame_binario)
                cv2.imshow('Mascara', mascara)
                cv2.imshow('AND', frame_and)

                if (cv2.waitKey(23)==27):
                    break
            
            video.release()
            cv2.destroyAllWindows()
        else:
            print('Can\'t Open Video ...')




if __name__=="__main__":
    cod = CodigoUN1()
    #cod.read_image()
    #cod.create_img()
    #cod.motion_detector()
    cod.seleccion_pixeles_rango()