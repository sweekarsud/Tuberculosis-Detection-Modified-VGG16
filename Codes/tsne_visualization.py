
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
from keras.preprocessing.image import img_to_array
from keras import activations

from vis.utils import utils

from sklearn.manifold import TSNE

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

layer_idx = utils.find_layer_idx(saved_model,'dense_1')

from keras import backend as K
def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
    activations = get_activations([X_batch,0])
    return activations

activations = get_activations(saved_model, layer_idx, X_VAL)

act = np.array(activations)

dimen = act.shape

print(dimen)

act = act.reshape(dimen[1],dimen[2]) 

tsne = TSNE(n_components=2, init='pca')

P1_tsne = tsne.fit_transform(act)

classes = ('Tuberculosis', 'Normal')
plt.figure(1)
colours = ListedColormap(['r','g'])
scatter = plt.scatter(P1_tsne[:,0], P1_tsne[:,1], c=Y_VAL, cmap=colours, marker='o')
plt.legend(handles=scatter.legend_elements()[0], labels=classes, fontsize='medium')
plt.legend(('Tuberculosis','Normal'))
plt.title('t-SNE: Baseline 1 - Fully Connected Layer')
plt.savefig('tSNE/t-SNE_base_fc1.png')


