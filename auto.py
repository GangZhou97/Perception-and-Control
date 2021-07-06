#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 16:05:55 2021

@author: zhougang
"""

from __future__ import print_function
import keras
import glob
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from zipfile import ZipFile
import matplotlib.pyplot as plt
# import sys
# sys.path.append('/home/zhougang/Environmental-perception/Environmental-classification/img_gray_auto1)
# import rotate
# import gray



def load_data(path = 'data/', num_classes = 6, image_shape = (100, 100, 1)):
    file_vec = glob.glob(path + '/*/*.png')
    if 0 == len(file_vec):
        with ZipFile('data.zip', 'r') as zipObj:
           # Extract all the contents of zip file in current directory
           zipObj.extractall()
        file_vec = glob.glob(path + '/*/*.png')
    file_num = len(file_vec)
    X = np.zeros((file_num,) + image_shape)
    y = np.zeros(file_num)
    idx = 0
    for n in range(num_classes):
        for file in glob.glob(path + str(n+1) + '/*.png'):
            img = cv2.imread(file, -1)
            img = np.reshape(img, image_shape)
            X[idx,...] = img
            y[idx] = n
            idx += 1
    y = keras.utils.to_categorical(y, num_classes=num_classes)
    print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=1)
    return (X_train, y_train), (X_test, y_test)


def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

def calc_net_size():
    sess = K.get_session()
    graph = sess.graph
    stats_graph(graph)



batch_size = 1
num_classes = 5
epochs = 30

image_shape = (100, 100, 1)

# input image dimensions
img_rows, img_cols = image_shape[0], image_shape[1]

(x_train, y_train), (x_test, y_test) = load_data(image_shape = image_shape, num_classes=num_classes)



if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# deep CNN
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


model_path = 'best_model.h5'
checkpoint = ModelCheckpoint(model_path, 
                             verbose=1, monitor='val_accuracy',
                             save_best_only=True, mode='auto')

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

is_train = False
if is_train:
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split=0.25, callbacks=[checkpoint])
# load the best model
model.load_weights( '/home/zhougang/Environmental-perception/Environmental-classification/checkpoint/best_model.h5')
# calc_net_size()

# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

def gray(source_data):
    x = source_data[:, 0]
    y = -source_data[:, 1]
    z = source_data[:, 2]
    
    
    y1 = -source_data[:, 1][np.logical_and(np.abs(x) < 0.01, np.abs(z) < 1)]
    z1 = source_data[:, 2][np.logical_and(np.abs(x) < 0.01, np.abs(z) < 1)]
    z2=(100*z1)-20
    y2=(100*y1)
    # y1 = -source_data[:, 1][np.logical_and(np.abs(x) < 0.01, np.abs(z) < 1.5,np.abs(z) >1)]
    # z1 = source_data[:, 2][np.logical_and(np.abs(x) < 0.01, np.abs(z) < 1.5,np.abs(z) >1)]
    # z2=(100*z1)-100
    # y2=(100*y1)+90




    img_original = np.zeros((100,100))

    for i in range(len(z2)-1):
       z_int = int(z2[i])    
       y_int = int(y2[i])
       img_original[z_int,y_int] = 1
    



    a =img_original
    mask = (a== 0).all(0)
    column_indices = np.where(mask)[0]
    a= a[:,~mask]
    
    b = a
    mask = (a== 0).all(0)
    column_indices = np.where(mask)[0]
    a= a[:,~mask]


    mask = (b == 0).all(1)
    column_indices = np.where(mask)[0]
    b = b[~mask,:]




    c = np.zeros((100,100))
    c[:b.shape[0], :b.shape[1]] = b
    return c
    
    
    
#将图片旋转一定角度
def rotate(img,angle,center=None,scale=1.0):
    
    (height, width) = img.shape[:2]
    
    if center is None:
        center = (width//2, height//2)
        
    rotate_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    
    img_rotated = cv2.warpAffine(img, rotate_matrix, (width, height))
    
    return img_rotated 

def change(n):
    n = n.reshape((1,n.shape[0],n.shape[1],-1))
    return n
    


def predict_frame(source_data):
    # source_data = np.load('//home/zhougang/Environmental-perception/image-ppt/1620286350.166.npy')[:,0:3]
    

    img = gray(source_data)
    img = rotate(img,90)
    cv2.imshow('binary image', img)
    cv2.waitKey(10)
    img = change(img)
    #cv2.imwrite('/home/zhougang/Environmental-perception/image/2.png', img*255)
   
    y = model.predict_classes(img)
    
    # img = np.zeros((80, 120, 3), np.uint8)
    # img.fill(90)
    # env_type_list = ['Groud', 'Up stairs','Down stairs','Up ramp','Down ramp']
    # text = env_type_list[y[0]]
    # cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    # cv2.imshow('Predict result', img)
    # cv2.waitKey(10)
    
    return y

def show(num):
    if num == 0:
     print ('平地')
    elif num == 1:
     print('上楼梯')
    elif num == 2:
     print('下楼梯')
    elif num == 3:
     print('上斜坡')
    elif num == 4:
     print('下斜坡')
    elif num == 5:
     print('障碍')
    
    return num
    




if __name__ == '__main__' :
    
    source_data = np.load('//home/zhougang/Environmental-perception/image-ppt/1620286350.166.npy')[:,0:3]
    predict_result = predict_frame(source_data)
    print(predict_result)
    
    
    
    
    