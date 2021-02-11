
import keras
from keras.preprocessing import image
from tqdm import tqdm
import cv2
import random
import re
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scipy.io as sio

from keras.models import load_model, Model
from keras import models


saved_model = load_model('mod2_vgg16.h5')

saved_model.summary()

import os
current_path = os.getcwd()
test_path = current_path + '/new_aug_dir/test_dir/*/*.png'

test_files = glob.glob(test_path)

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

X_VAL, Y_VAL = process_image(x_Val, y_Val)

Y_VAL_wocat = Y_VAL

Y_VAL = keras.utils.to_categorical(Y_VAL)

Y_VAL_pred = saved_model.predict(X_VAL)

from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_VAL_wocat, Y_VAL_pred.argmax(axis=1))

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras)
plt.axis([0,1,0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('roc_curve.png')

precision, recall, _ = precision_recall_curve(Y_VAL_wocat, Y_VAL_pred.argmax(axis=1))

print(precision)
print(Y_VAL_pred.argmax(axis=1))

no_skill = len(Y_VAL_wocat[Y_VAL_wocat==1]) / len(Y_VAL_wocat)
plt.figure(2)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--')
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig('prec_rec.png')
