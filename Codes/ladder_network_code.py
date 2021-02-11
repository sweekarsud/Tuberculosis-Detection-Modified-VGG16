from __future__ import print_function

from keras.datasets import mnist
import keras
import random
import cv2
import numpy
import re
import glob
from tqdm import tqdm
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf

from ladder_net import get_ladder_network_fc

# get the dataset
inp_size = 128*128 # size of mnist dataset 
n_classes = 2

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
            labels.append(0)
        else:
            labels.append(1)
    return(all_files, labels)


x_Train, y_Train = extract_label(new_train_files)
x_Test, y_Test = extract_label(new_test_files)    

def process_image(imagepaths, ylabel):
    
    X = []
    Y = []
    for path in imagepaths:
        img = cv2.imread(path) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (128, 128)) 
        X.append(img)
        
    for label in ylabel:
        Y.append(int(label))
    
    X = np.array(X, dtype="uint8")
    X = X.reshape(len(imagepaths), 128*128) 
    Y = np.array(Y)

    print("Images loaded: ", len(X))
    print("Labels loaded: ", len(Y))

    
    return(X, Y)


### Creating proper sets
x_train, y_train = process_image(x_Train, y_Train)
x_test, y_test = process_image(x_Test, y_Test)


x_train = x_train / 255.0
x_test = x_test / 255.0
y_train_wocat = y_train
y_train = keras.utils.to_categorical(y_train)
y_test_wocat = y_test
y_test = keras.utils.to_categorical(y_test)


idxs_annot = range(x_train.shape[0])
random.seed(0)
idxs_annot = np.random.choice(x_train.shape[0], len(x_train))

x_train_unlabeled = x_train
x_train_labeled   = x_train[idxs_annot]
y_train_labeled   = y_train[idxs_annot]

n_rep = x_train_unlabeled.shape[0] // x_train_labeled.shape[0]
x_train_labeled_rep = np.concatenate([x_train_labeled]*n_rep)
y_train_labeled_rep = np.concatenate([y_train_labeled]*n_rep)


# initialize the model 
model = get_ladder_network_fc()
model.summary()


for i in range(100):
    history = model.fit([x_train_labeled_rep, x_train_unlabeled], y_train_labeled_rep, epochs=1)
    print('Epoch #',i);

