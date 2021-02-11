
from keras.models import load_model

saved_model = load_model('mod2_vgg16.h5')

saved_model.summary()

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from vis.visualization import visualize_saliency
from vis.utils import utils
from keras import activations
import matplotlib.image as mpimg
import scipy.ndimage as ndimage

import os
current_path = os.getcwd()
test_path = current_path + '/new_aug_dir/train_dir/Tuberculosis/CHNCXR_0500_1.png'

img = mpimg.imread(test_path)

layer_idx = utils.find_layer_idx(saved_model,'dense_2')
saved_model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(saved_model)

grads = visualize_saliency(model, layer_idx, filter_indices=None, seed_input=img, backprop_modifier=None, grad_modifier="absolute")

gaus = ndimage.gaussian_filter(grads[:,:], sigma=5)

plt.imshow(img)
plt.imshow(gaus,alpha=0.5)
plt.savefig('res_salvg_5.png')


