# -*- coding:utf-8 -*-
import cv2
import random
import time
import ctypes

from train import Model
from input import resize_with_pad, write_image
from input import IMAGE_SIZE, GRAY_MODE

DEBUG_OUTPUT = True
CropPadding = 10

StrictMode = False
MaxPromptDelay = 500  # in microsecond
MaxFailDelay = 3000 # in microsecond
SampleInterval = 100 # in microsecond

cascade_path = "F:/Software/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml"

def extendFaceRect(rect):
    [x, y, w, h] = rect
    if y > CropPadding: y = y - CropPadding
    else: y = 0
    h += 2*CropPadding
    if x > CropPadding: x = x - CropPadding
    else: x = 0
    w += 2*CropPadding
    return [x, y, w, h]

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    model = Model()
    model.load()

    # Get Cascade Classifier
    cascade = cv2.CascadeClassifier(cascade_path)

    isme=0
    notme=0

    nDelay = 0

    # Run window in other thread
    cv2.startWindowThread()

    while True:
        _, frame = cap.read()

        # To gray image
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Recognize faces
        facerect = cascade.detectMultiScale(
            frame_gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(10, 10),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        recStatus = 0
        if len(facerect) > 0:
            print('Face detected.')
            color = (255, 255, 255)  # 白

            if nDelay >= MaxPromptDelay: # Show the recognize windows
                for (x, y, w, h) in facerect:
                    [x, y, w, h] = extendFaceRect([x, y, w, h])
                    buffer = frame.copy()
                    cv2.rectangle(buffer, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                cv2.imshow('Recognizing', buffer)
                #cv2.namedWindow('Recognizing', cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

            for rect in facerect:
                [x, y, width, height] = extendFaceRect(rect)
                
                # Crop the face
                if GRAY_MODE == True:
                    img_predict = frame_gray[y: y + height, x: x + width]
                else:
                    img_predict = frame[y: y + height, x: x + width]

                # Predict face
                if GRAY_MODE == True:
                    result = model.predict(img_predict, img_channels=1)
                else:
                    result = model.predict(img_predict)

                if DEBUG_OUTPUT == True:
                    outimg = frame[y: y + height, x: x + width]
                    if result == 0:
                        write_image('./output/isme/' + str(random.randint(1,999999)) + '.jpg', outimg)
                    else:
                        write_image('./output/notme/' + str(random.randint(1,999999)) + '.jpg', outimg)

                if result == 0:  # Is me
                    print("It's you! Donny!")
                    isme+=1
                    recStatus = 1
                else:
                    print('Not Donny.')
                    notme+=1
                    if recStatus == 0:
                        recStatus = -1

                print('isme', isme, 'notme', notme)

        # End if Face Detected
        
        print(nDelay)

        if recStatus == -1 or (recStatus == 0 and (StrictMode or nDelay >= MaxPromptDelay)):
            nDelay += SampleInterval
        elif recStatus == 1:
            nDelay = 0
            cv2.destroyWindow('Recognizing')

        if nDelay >= MaxFailDelay: # Lock Windows
            ctypes.windll.user32.LockWorkStation()
            cv2.destroyWindow('Recognizing')
            
        cv2.waitKey(1)

        time.sleep(SampleInterval/1000)

    # End while True

    # Stop recognize
    cap.release()
    cv2.destroyAllWindows()
