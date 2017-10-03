# -*- coding: utf-8 -*-
import os

import numpy as np
import cv2

IMAGE_SIZE = 64
GRAY_MODE = True # Transit images to grayscale images
DEBUG_OUTPUT = False # Output processed images
DEBUG_VERBOSE = False # Print more detail
EX_DATA = True # Label images by the name of the first subfolder

def resize_with_pad(image, height=IMAGE_SIZE, width=IMAGE_SIZE):

    def get_padding_size(image):
        if GRAY_MODE:
            h, w = image.shape
        else:
            h, w, _ = image.shape
        longest_edge = max(h, w)
        top, bottom, left, right = (0, 0, 0, 0)
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(image)
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    resized_image = cv2.resize(constant, (height, width))

    return resized_image


def traverse_dir(path, images=[], labels=[]):
    for file_or_dir in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))
        
        if DEBUG_VERBOSE == True:
            print(abs_path)

        if os.path.isdir(abs_path):  # dir
            traverse_dir(abs_path, images, labels)
        else:                        # file
            if file_or_dir.endswith('.jpg'):
                image = read_image(abs_path)
                if DEBUG_OUTPUT == True:
                    write_image('./output/' + path.split('\\')[-1] + os.path.basename(abs_path), image)
                images.append(image)
                labels.append(path)

    return images, labels


def read_image(file_path):
    image = cv2.imread(file_path)
    if GRAY_MODE == True:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)

    return image

def write_image(file_path, image):
    print('outputing: ', file_path)
    cv2.imwrite(file_path, image)

def extract_data(path):
    if EX_DATA == False:
        images, labels = traverse_dir(path)
        images = np.array(images)
        labels = np.array([0 if label.endswith('me') else 1 for label in labels])
    else:
        # Label by the name of the first folder's
        imagesMe, labelsMe = traverse_dir(path + '/me')
        imagesMe = np.array(imagesMe)
        labelsMe = np.array([0 for _ in imagesMe])
        imagesOther, labelsOther = traverse_dir(path + '/other', [], [])
        imagesOther = np.array(imagesOther)
        labelsOther = np.array([1 for _ in imagesOther])
        images = np.concatenate((imagesMe, imagesOther))
        labels = np.concatenate((labelsMe, labelsOther))

    return images, labels
