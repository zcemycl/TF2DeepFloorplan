import numpy as np

import tensorflow as tf

#from scipy.misc import imread, imresize, imsave
from skimage.io import imread, imsave
from skimage.transform import resize as imresize
from matplotlib import pyplot as plt
from utils.rgb_ind_convertor import *

import os
import sys
import glob 
import time

#path = '/d2/studies/TF2DeepFloorplan/dataset/r3d_test.txt'


def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- *
# ------------------------------------------------------------------------------------------------------------------------------------- *
# Following are only for multi-task network. Two labels(boundary and room.)

def load_bd_rm_images(path):
    paths = path.split('\t')
    fname = paths[0].split('/')[-1].split('.')[0]
    image = imread(paths[0], pilmode='RGB')
    close = imread(paths[2], pilmode='L')
    room  = imread(paths[3], pilmode='RGB')
    close_wall = imread(paths[4], pilmode='L')

	# NOTE: imresize will rescale the image to range [0, 255], also cast data into uint8 or uint32
    image = imresize(image, (512, 512, 3), mode='constant', cval=0, preserve_range=True)
    close = imresize(close, (512, 512), mode='constant', cval=0, preserve_range=True) #/ 255. #added preserve_range for TFR4
    close_wall = imresize(close_wall, (512, 512), mode='constant', cval=0, preserve_range=True) # / 255. #Added preserve_range for TFR4
    room = imresize(room, (512, 512, 3), mode='constant', cval=0, preserve_range=True)

    room_ind = rgb2ind(room)

	# merge result
    d_ind = (close>0.5).astype(np.uint8)#Changed close>0.5 to close>(255/2) for TFRecords_4
    cw_ind = (close_wall>0.5).astype(np.uint8)#Changed close>0.5 to close>(255/2) for TFRecords_4

    cw_ind[cw_ind==1] = 2
    cw_ind[d_ind==1] = 1
    
	# make sure the dtype is uint8
    image = image.astype(np.uint8)
    room_ind = room_ind.astype(np.uint8)
    cw_ind = cw_ind.astype(np.uint8)

	# debugging
	# merge = ind2rgb(room_ind, color_map=floorplan_fuse_map)
	# rm = ind2rgb(room_ind)
	# bd = ind2rgb(cw_ind, color_map=floorplan_boundary_map)
	# plt.subplot(131)
	# plt.imshow(image)
	# plt.subplot(132)
	# plt.imshow(rm/256.)
	# plt.subplot(133)
	# plt.imshow(bd/256.)
	# plt.show()

    return image, cw_ind, room_ind, d_ind, fname

def write_bd_rm_record(paths, name='dataset.tfrecords'):
	writer = tf.compat.v1.python_io.TFRecordWriter(name)
	
	for i in range(len(paths)):
		# Load the image
		image, cw_ind, room_ind, d_ind, fname = load_bd_rm_images(paths[i])

		# Create a feature
		feature = {'image': _bytes_feature(tf.compat.as_bytes(image.tobytes())),#changed tostring() to tobytes()
					'boundary': _bytes_feature(tf.compat.as_bytes(cw_ind.tobytes())),
					'room': _bytes_feature(tf.compat.as_bytes(room_ind.tobytes())),
					'door': _bytes_feature(tf.compat.as_bytes(d_ind.tobytes())),
                    'zname': _bytes_feature(tf.compat.as_bytes(fname))}#name
		
		# Create an example protocol buffer
		example = tf.train.Example(features=tf.train.Features(feature=feature))
    
		# Serialize to string and write on the file
		writer.write(example.SerializeToString())
    
	writer.close()

def read_bd_rm_record(data_path, batch_size=1, size=512):
    feature = {'image': tf.FixedLenFeature(shape=(), dtype=tf.string),
				'boundary': tf.FixedLenFeature(shape=(), dtype=tf.string),
				'room': tf.FixedLenFeature(shape=(), dtype=tf.string),
				'door': tf.FixedLenFeature(shape=(), dtype=tf.string),
                'zname': name}#tf.FixedLenFeature(shape=(), dtype=tf.string)

	# Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=None, shuffle=False, capacity=batch_size*128)
	
	# Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

	# Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)

	# Convert the image data from string back to the numbers
    image = tf.decode_raw(features['image'], tf.uint8)
    boundary = tf.decode_raw(features['boundary'], tf.uint8)
    room = tf.decode_raw(features['room'], tf.uint8)
    door = tf.decode_raw(features['door'], tf.uint8)
   # name = tf.decode_raw(features['name'], tf.string)
	# Cast data
    image = tf.cast(image, dtype=tf.float32)

	# Reshape image data into the original shape
    image = tf.reshape(image, [size, size, 3])
    boundary = tf.reshape(boundary, [size, size])
    room = tf.reshape(room, [size, size])
    door = tf.reshape(door, [size, size])

	# Any preprocessing here ...
	# normalize 
    image = tf.divide(image, tf.constant(255.0))

	# Genereate one hot room label
    label_boundary = tf.one_hot(boundary, 3, axis=-1)
    label_room = tf.one_hot(room, 9, axis=-1)

	# Creates batches by randomly shuffling tensors
    images, label_boundaries, label_rooms, label_doors, names = tf.train.shuffle_batch([image, label_boundary, label_room, door, name], 
						batch_size=batch_size, capacity=batch_size*128, num_threads=1, min_after_dequeue=batch_size*32)	

	# images, walls = tf.train.shuffle_batch([image, wall], 
						# batch_size=batch_size, capacity=batch_size*128, num_threads=1, min_after_dequeue=batch_size*32)	

    return {'images': images, 'label_boundaries': label_boundaries, 'label_rooms': label_rooms, 'label_doors': label_doors, 'names': names}
