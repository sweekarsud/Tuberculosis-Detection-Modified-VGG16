# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 01:10:08 2020

@author: Sweekar
"""


from keras.models import *
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import keras
from keras.metrics import top_k_categorical_accuracy
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers import BatchNormalization, MaxPooling2D, AveragePooling2D, Conv2D
from keras.layers import Flatten, GlobalAveragePooling2D
from keras import optimizers
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler, EarlyStopping, CSVLogger, ReduceLROnPlateau, TensorBoard
from keras.utils import np_utils
from keras.utils import plot_model
from keras.models import Model
from keras.applications  import MobileNetV2, VGG19, VGG16, ResNet50, InceptionV3
from sklearn.metrics import confusion_matrix

from tqdm import tqdm
import cv2
import random
import re
import glob
import numpy as np


########## VGG 16 Pre-Trained ##########
input_shape = (224,224,3)

#########################################

import os
current_path = os.getcwd()
train_path = current_path + '/new_aug_dir/train_dir/*/*.png'
test_path = current_path + '/new_aug_dir/test_dir/*/*.png' 

train_files = glob.glob(train_path)
test_files = glob.glob(test_path)


new_train_files = random.sample(train_files, len(train_files))
new_test_files = random.sample(test_files, len(test_files))

def extract_label(all_files):
    labels = []
    for file in tqdm(all_files):
        current_label = re.findall('Normal', file)
        if current_label == []:
            labels.append(1)
        else:
            labels.append(0)
    return(all_files, labels)

x_Train, y_Train = extract_label(new_train_files)
x_Val, y_Val = extract_label(new_test_files)


def process_image(imagepaths, ylabel):

    X = []
    Y = []
    for path in imagepaths:
        img = cv2.imread(path) # Reads image and returns np.array
        X.append(img)

    for label in ylabel:
        Y.append(int(label))

    X = np.array(X, dtype="uint8")
    X = X.reshape(len(imagepaths), 224, 224, 3) # Needed to reshape so CNN knows it's different images
    Y = np.array(Y)

    print("Images loaded: ", len(X))
    print("Labels loaded: ", len(Y))


    return(X, Y)

### Creating proper sets
X_TRAIN, Y_TRAIN = process_image(x_Train, y_Train)
X_VAL, Y_VAL = process_image(x_Val, y_Val)

Y_TRAIN_wocat = Y_TRAIN
Y_VAL_wocat = Y_VAL

base_model = VGG16(weights = 'imagenet', include_top=False, input_shape=(224, 224, 3)) 


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)


model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.00008), metrics=['accuracy'])

history = model.fit(X_TRAIN, Y_TRAIN, validation_data=(X_VAL, Y_VAL), batch_size=8, epochs=15, verbose=1)

