import random
import os
import numpy as np

from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K

from input import extract_data, resize_with_pad, IMAGE_SIZE, GRAY_MODE

DEBUG_MUTE = True

class DataSet(object):

    TRAIN_DATA = './data/train/'

    def __init__(self):
        self.X_train = None
        self.X_valid = None
        self.X_test = None
        self.Y_train = None
        self.Y_valid = None
        self.Y_test = None

    def read(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE, img_channels=3, nb_classes=2):
        
        images, labels = extract_data(self.TRAIN_DATA)
        labels = np.reshape(labels, [-1])
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=random.randint(0, 100))
        X_valid, X_test, y_valid, y_test = train_test_split(images, labels, test_size=0.5, random_state=random.randint(0, 100))
        if K.image_dim_ordering() == 'th':
            X_train = X_train.reshape(X_train.shape[0], img_channels, img_rows, img_cols)
            X_valid = X_valid.reshape(X_valid.shape[0], img_channels, img_rows, img_cols)
            X_test = X_test.reshape(X_test.shape[0], img_channels, img_rows, img_cols)
            input_shape = (img_channels, img_rows, img_cols)
        else:
            X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_channels)
            X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, img_channels)
            X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_channels)
            input_shape = (img_rows, img_cols, img_channels)

        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_valid.shape[0], 'valid samples')
        print(X_test.shape[0], 'test samples')

        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_valid = np_utils.to_categorical(y_valid, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)

        X_train = X_train.astype('float32')
        X_valid = X_valid.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_valid /= 255
        X_test /= 255

        self.X_train = X_train
        self.X_valid = X_valid
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_valid = Y_valid
        self.Y_test = Y_test


class Model(object):

    FILE_PATH = './store/model.h5'

    # DropoutWeights = [ 0.25, 0.25, 0.5 ]
    DropoutWeights = [ 0.1, 0.1, 0.1, 0.2 ]

    # TrainEpoch = 10
    TrainEpoch = 320
    # enough: 640; total fit: 800

    def __init__(self):
        self.model = None

    def build_model(self, dataset, nb_classes=2):
        self.model = Sequential()

        self.model.add(Convolution2D(32, 3, 3, border_mode = 'same', input_shape = dataset.X_train.shape[1:]))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(32, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(self.DropoutWeights[0]))
        # self.model.add(Dropout(0.15))

        self.model.add(Convolution2D(64, 3, 3, border_mode = 'same'))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(self.DropoutWeights[1]))
        # self.model.add(Dropout(0.15))

        self.model.add(Convolution2D(64, 3, 3, border_mode = 'same'))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(self.DropoutWeights[2]))
        # self.model.add(Dropout(0.15))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(self.DropoutWeights[3]))
        # self.model.add(Dropout(0.35))
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax'))

        self.model.summary()

    def train(self, dataset, batch_size=32, nb_epoch=40, data_augmentation=True):
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                            optimizer=sgd,
                            metrics=['accuracy'])
        if not data_augmentation:
            print('Not using data augmentation.')
            self.model.fit(dataset.X_train, dataset.Y_train,
                            batch_size=batch_size,
                            nb_epoch=nb_epoch,
                            validation_data=(dataset.X_valid, dataset.Y_valid),
                            shuffle=True)
        else:
            print('Using real-time data augmentation.')

            datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=False
            )

            datagen.fit(dataset.X_train)

            self.model.fit_generator(
                datagen.flow(dataset.X_train, dataset.Y_train, batch_size=batch_size),
                samples_per_epoch=dataset.X_train.shape[0],
                nb_epoch=nb_epoch,
                validation_data=(dataset.X_valid, dataset.Y_valid)
            )

    def save(self, file_path=FILE_PATH):
        file_dir = os.path.dirname(file_path)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        self.model.save(file_path)
        print('Model Saved.')

    def load(self, file_path=FILE_PATH):
        self.model = load_model(file_path)
        print('Model Loaded.')

    def predict(self, image, img_channels=3):
        if K.image_dim_ordering() == 'th' and image.shape != (1,img_channels, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_with_pad(image)
            image = image.reshape((1, img_channels, IMAGE_SIZE, IMAGE_SIZE))
        elif K.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, img_channels):
            image = resize_with_pad(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, img_channels))
        image = image.astype('float32')
        image /= 255
        if DEBUG_MUTE:
            result = self.model.predict_proba(image, verbose=0)
            result = self.model.predict_classes(image, verbose=0)
        else:
            result = self.model.predict_proba(image)
            print(result)
            result = self.model.predict_classes(image)
            print(result)

        return result[0]

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.X_test, dataset.Y_test, verbose=0)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1]*100))


if __name__ == '__main__':
    dataset = DataSet()
    if GRAY_MODE:
        dataset.read(img_channels=1)
    else:
        dataset.read()

    model = Model()
    #model.load()
    model.build_model(dataset)
    model.train(dataset, nb_epoch=model.TrainEpoch)
    model.save()

    model = Model()
    model.load()
    model.evaluate(dataset)

