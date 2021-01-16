import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pdb
import os

def convert_one_hot_to_image(one_hot,dtype='float',act=None):
    if act=='softmax':
        one_hot = tf.keras.activations.softmax(one_hot)
    [n,h,w,c] = one_hot.shape.as_list()
    im=tf.reshape(tf.keras.backend.argmax(one_hot,axis=-1),
                  [n,h,w,1])
    if dtype=='int':
        im = tf.cast(im,dtype=tf.uint8)
    else:
        im = tf.cast(im,dtype=tf.float32)
    return im

def _parse_function(example_proto):
    feature = {'image':tf.io.FixedLenFeature([],tf.string),
              'boundary':tf.io.FixedLenFeature([],tf.string),
              'room':tf.io.FixedLenFeature([],tf.string),
              'door':tf.io.FixedLenFeature([],tf.string),
              'zname':tf.io.FixedLenFeature([], tf.string)}
    return tf.io.parse_single_example(example_proto,feature)

def decodeAllRaw(x):
    image = tf.io.decode_raw(x['image'],tf.uint8)
    boundary = tf.io.decode_raw(x['boundary'],tf.uint8)
    room = tf.io.decode_raw(x['room'],tf.uint8)
    return image,boundary,room

def preprocess(img,bound,room,size=512):
    img = tf.cast(img,dtype=tf.float32)/255
    img = tf.reshape(img,[-1,size,size,3])
    bound = tf.reshape(bound,[-1,size,size])
    room = tf.reshape(room,[-1,size,size])
    hot_b = tf.one_hot(bound,3,axis=-1)
    hot_r = tf.one_hot(room,9,axis=-1)
    return img,bound,room,hot_b,hot_r

def decodeRawInfer(x):
    image = tf.io.decode_raw(x['image'],tf.uint8)
    return image

def preprocessInfer(img,size=512):
    img = tf.cast(img,dtype=tf.float32)/255
    img = tf.reshape(img,[-1,size,size,3])
    return img


def loadDataset(size=512, train=True):
    if train:
        raw_dataset = tf.data.TFRecordDataset(os.path.join(os.getcwd(),'dataset/NY_train_withNames_5.tfrecords'))
    elif not train:
        raw_dataset = tf.data.TFRecordDataset(os.path.join(os.getcwd(),'dataset/NY_test_withNames_5.tfrecords'))
    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset

def identify_bounds(bound):
    return bound==1

#def loadImages(directory, file_format='.jpg', size=512):
#    filename_queue = tf.train.string_input_producer(
#        tf.train.match_filenames_once(os.path.join(directory, '*' + file_format)))
#    files = os.listdir(directory)
#    images=[]
#    for f in files:
#        if f.endswith(file_format):
#            fullpath=os.path.join(directory, f)
#            images.append(fullpath)
            

if __name__ == "__main__":
    dataset = loadDataset()
    for ite in range(2):
        for data in list(dataset.shuffle(400).batch(1)):
            img,bound,room = decodeAllRaw(data)
            img,bound,room,hb,hr = preprocess(img,bound,room)
            plt.subplot(1,3,1);plt.imshow(img[0].numpy())
            plt.subplot(1,3,2);plt.imshow(bound[0].numpy())
            plt.subplot(1,3,3);plt.imshow(convert_one_hot_to_image(hb)[0].numpy());plt.show()


            pdb.set_trace()    
            break


    
