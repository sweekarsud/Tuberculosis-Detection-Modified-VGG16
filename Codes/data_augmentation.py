"""
Created on Sun Mar 15 17:52:40 2020

@author: Sweekar
"""

import pandas as pd
import numpy as np
import os
import cv2
import imageio

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import shutil
from keras.preprocessing.image import ImageDataGenerator

NUM_AUG_IMAGES_WANTED = 4000 

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

current_path = os.getcwd()

shenz_datapath = current_path + '/proc_data/ChinaSet_AllFiles'
montg_datapath = current_path + '/proc_data/MontgomerySet'

shen_image_list = os.listdir(shenz_datapath)
mont_image_list = os.listdir(montg_datapath)

df_shen = pd.DataFrame(shen_image_list, columns=['image_path'])
df_mont = pd.DataFrame(mont_image_list, columns=['image_path'])

df_shen = df_shen[df_shen['image_path'] != 'Thumbs.db']
df_mont = df_mont[df_mont['image_path'] != 'Thumbs.db']

df_shen.reset_index(inplace=True, drop=True)
df_mont.reset_index(inplace=True, drop=True)

'''
Label Assignment
'''
def extract_target(x):
    target = int(x[-5])
    if target == 0:
        return 'Normal'
    if target == 1:
        return 'Tuberculosis'

'''
pandas style of passing to function using apply
'''    
df_shen['target'] = df_shen['image_path'].apply(extract_target)
df_mont['target'] = df_mont['image_path'].apply(extract_target)

    
shen_img_path = shenz_datapath + '/' 
mont_img_path = montg_datapath + '/' 

# combining and shuffling the two data
df_data = pd.concat([df_shen, df_mont], axis=0).reset_index(drop=True)
df_data = shuffle(df_data)

df_data['labels'] = df_data['target'].map({'Normal':0, 'Tuberculosis':1})

y = df_data['labels']
df_train, df_test = train_test_split(df_data, test_size=0.30, random_state=2, stratify=y)

'''
Data Augmentation:
'''
base_dir = 'new_aug_dir'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train_dir')
os.mkdir(train_dir)

test_dir = os.path.join(base_dir, 'test_dir')
os.mkdir(test_dir)

Normal = os.path.join(train_dir, 'Normal')
os.mkdir(Normal)
Tuberculosis = os.path.join(train_dir, 'Tuberculosis')
os.mkdir(Tuberculosis)

Normal = os.path.join(test_dir, 'Normal')
os.mkdir(Normal)
Tuberculosis = os.path.join(test_dir, 'Tuberculosis')
os.mkdir(Tuberculosis)

folder_1 = os.listdir(shen_img_path)
folder_2 = os.listdir(mont_img_path)

train_list = list(df_train['image_path'])
test_list = list(df_test['image_path'])

comp_path = current_path + '/'

df_data.set_index('image_path', inplace=True)

for image in train_list:  
    fname = image
    label = df_data.loc[image, 'target']
    
    if fname in folder_1:
        
        src = shen_img_path + fname
        dst = comp_path + train_dir + '/' + label + '/' + fname
        
        image = cv2.imread(src)
        image = cv2.resize(image, (224, 224))
        cv2.imwrite(dst, image)
        
    if fname in folder_2:
        
        src = mont_img_path + fname
        dst = comp_path + train_dir + '/' + label + '/' + fname
        
        image = cv2.imread(src)
        image = cv2.resize(image, (224, 224))
        cv2.imwrite(dst, image)
        
    
for image in test_list:  
    fname = image
    label = df_data.loc[image,'target']
    
    if fname in folder_1:
        
        src = shen_img_path + fname
        dst = comp_path + test_dir + '/' + label + '/' + fname
        
        image = cv2.imread(src)
        image = cv2.resize(image, (224, 224))
        cv2.imwrite(dst, image)
        
    if fname in folder_2:
        
        src = mont_img_path + fname
        dst = comp_path + test_dir + '/' + label + '/' + fname
        
        image = cv2.imread(src)
        image = cv2.resize(image, (224, 224))
        cv2.imwrite(dst, image)
    
    
print('Data Augmentation is starting...')    
'''
Data Augmentation
'''     
    
class_list = ['Normal','Tuberculosis']

for item in class_list:
    
    aug_dir = 'aug_dir'
    os.mkdir(aug_dir)
    
    img_dir = os.path.join(aug_dir, 'img_dir')
    os.mkdir(img_dir)
        
    img_class = item

    img_list = os.listdir(comp_path + 'new_aug_dir/train_dir/' + img_class)

    for fname in img_list:
            src = os.path.join(comp_path + 'new_aug_dir/train_dir/' + img_class, fname)
            dst = os.path.join(img_dir, fname)
            shutil.copyfile(src, dst)

    path = aug_dir
    save_path = comp_path + 'new_aug_dir/train_dir/' + img_class

    datagen = ImageDataGenerator(rescale=1/255, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')

    batch_size = 20
    
    aug_datagen = datagen.flow_from_directory(path, save_to_dir=save_path, save_format='png', target_size=(IMAGE_HEIGHT,IMAGE_WIDTH), batch_size=batch_size)
    
    num_files = len(os.listdir(img_dir))
    
    num_batches = int(np.ceil((NUM_AUG_IMAGES_WANTED-num_files)/batch_size))

    for i in range(0,num_batches):
        imgs, labels = next(aug_datagen)
    
    shutil.rmtree('aug_dir')
  
    
