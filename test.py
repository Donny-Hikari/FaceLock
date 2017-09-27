# -*- coding:utf-8 -*-
import cv2
import random
import time

from train import Model
from input import resize_with_pad, write_image
from input import IMAGE_SIZE, GRAY_MODE
# from image_show import show_image

DEBUG_OUTPUT = True
CropPadding = 10

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

    # カスケード分類器の特徴量を取得する
    # Get Cascade Classifier
    cascade = cv2.CascadeClassifier(cascade_path)

    isme=0
    notme=0

    while True:
        _, frame = cap.read()

        # グレースケール変換
        # To gray image
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 物体認識（顔認識）の実行
        # Recognize faces
        facerect = cascade.detectMultiScale(
            frame_gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(10, 10),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        #facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.01, minNeighbors=3, minSize=(3, 3))
        if len(facerect) > 0:
            print('face detected')
            color = (255, 255, 255)  # 白

            for (x, y, w, h) in facerect:
                [x, y, w, h] = extendFaceRect([x, y, w, h])
                buffer = frame.copy()
                cv2.rectangle(buffer, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            cv2.imshow('Recognizing', buffer)

            for rect in facerect:
                # 検出した顔を囲む矩形の作成
                #cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), color, thickness=2)

                # x, y = rect[0:2]
                # width, height = rect[2:4]
                [x, y, width, height] = extendFaceRect(rect)
                
                # Crop the face
                if GRAY_MODE == True:
                    img_predict = frame_gray[y: y + height, x: x + width]
                else:
                    img_predict = frame[y: y + height, x: x + width]
                    # frame = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

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

                #if result[0] > 0.70 :
                #    result = 0
                #else:
                #    result = 1

                if result == 0:  # Is me
                    print("It's you! Donny!")
                    isme+=1
                else:
                    print('Not Donny.')
                    notme+=1

                print('isme', isme, 'notme', notme)

        # Wait for input
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        #10msecキー入力待ち
        #k = cv2.waitKey(0)
        #Escキーを押されたら終了
        #if k == 27:
        #    break

    #キャプチャを終了
    # Stop Recognize
    cap.release()
    cv2.destroyAllWindows()
