#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 12:10:34 2021

@author: smith
"""

import os
os.chdir('/d2/studies/TF2DeepFloorplan/')
import tensorflow as tf
import io
import tqdm
from net import *
from loss import *
from data import *
import argparse
import pandas as pd
from PIL import Image
from datetime import datetime
from skimage.io import imread, imsave
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import train


config = {'batchsize':1, 'lr':1e-4, 'wd':1e-5, 'epochs':100, 'logdir':'./log/Jan13', 'saveTensorInterval':10,
         'saveModelInterval':20, 'restore':None, 'outdir':'./outJan13', 'train':True}
data = list(dataset.shuffle(400).batch(config['batchsize']))[0]

raw_dataset = tf.data.TFRecordDataset('/d2/studies/TF2DeepFloorplan/dataset/NY_train_withNames_3.tfrecords')
parsed_dataset = raw_dataset.map(_parse_function)

data = list(parsed_dataset.batch(config['batchsize']))[0]


def _parse_function(example_proto):
    feature = {'image':tf.io.FixedLenFeature([],tf.string),
              'boundary':tf.io.FixedLenFeature([],tf.string),
              'room':tf.io.FixedLenFeature([],tf.string),
              'door':tf.io.FixedLenFeature([],tf.string),
              'zname':tf.io.FixedLenFeature([], tf.string)}
    return tf.io.parse_single_example(example_proto,feature)




dataset = loadDataset(train=config['train'])


data_path = '/d2/studies/TF2DeepFloorplan/MT_trainingData'

read_seg_record(data_path, batch_size=1, size=512)

train.main(config)

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


embed_file='/d2/studies/TF2DeepFloorplan/dataset/NY_train_withNames_2.tfrecords'
sample=5

dataset = tf.data.TFRecordDataset(embed_file)
for record in dataset.take(sample).map(_parse_example):
  print("{}: {}".format(record['text'].numpy().decode('utf-8'), record['embedding'].numpy()[:10]))


feature = {'image':tf.io.FixedLenFeature([],tf.string),
          'boundary':tf.io.FixedLenFeature([],tf.string),
          'room':tf.io.FixedLenFeature([],tf.string),
          'door':tf.io.FixedLenFeature([],tf.string),
          'zname':tf.io.FixedLenFeature([], tf.string)}

def _parse_function(example_proto):
    feature = {'image':tf.io.FixedLenFeature([],tf.string),
              'boundary':tf.io.FixedLenFeature([],tf.string),
              'room':tf.io.FixedLenFeature([],tf.string),
              'door':tf.io.FixedLenFeature([],tf.string),
              'zname':tf.io.FixedLenFeature([], tf.string)}
    return tf.io.parse_single_example(example_proto,feature)

for record in dataset.take(sample).map(_parse_function):
  print("{}: {}".format(record['zname'].numpy().decode('utf-8'), record['image'].numpy()[:10]))


