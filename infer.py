#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 11:34:09 2021

@author: smith
"""

import os
#os.chdir('/d2/studies/TF2DeepFloorplan')
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
from skimage import img_as_float
import matplotlib.pyplot as plt
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def init(config):
    """Initialize the model (via net.py), load the data (via data.py) using a Keras Adam optimizer.
    You can see other optimizer options you may want to try out at: https://keras.io/api/optimizers/.
    The SGD optimizer is probably the most commonly used, though it seems more modern models have been
    switching to Adam. The way it is now is probably ideal but this is something you can play with or discuss
    in your write-up.
    
    Parameters
    ----------
    config : dict or argparse namespace
        Dictionary of {'batchsize':2, 'lr':1e-4, 'wd':1e-5, 
                       'epochs':1000, 'logdir':'./log/store', 
                       'saveTensorInterval':10, 'saveModelInterval':2, 
                       'restore':'./log/store/', 'outdir':'./out'}

    Returns
    -------
    dataset : Tensorflow dataset.
    model : Tensorflow model.
    optim : Keras optimizer.

    """
   # if config['restore'] is not None:
   #     print("Loading pre-trained model from {}".format(config['logdir']))
   #     model=load_model(config['restore'])
    #else:
    model = deepfloorplanModel()
    dataset = loadDataset(train=config['train'])
    #optim = tf.keras.optimizers.AdamW(learning_rate=config.lr,weight_decay=config.wd)
    optim = tf.keras.optimizers.Adam(learning_rate=config['lr'])
    return dataset,model,optim

def plot_to_image(figure, pltiter, name, directory, save=False):
    """Convert tensors to images.
    
    Parameters
    ----------
    figure : tensor
        Tensorflow tensor to convert to image.
    pltiter : int
        Iteration to plot.
    directory : string
        Output directory.
    """
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    imSave = Image.open(buf)
    if save:
        imSave.save(os.path.join(directory, str(pltiter) + '/' + name + '.png'))
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

def infer_image_grid(img,logr,logcw, pltiter, name, outdir):
    """Make figure with 4 subplots containing training data & results.
    
    Parameters
    ----------
    img : Tensor image.
        Tensor of true image to plot
    bound : Tensor image.
        True Room boundaries
    room: Tensor image.
        Room layout
    logr: Tensor image.
        Predicted boundaries.
    logcw: Tensor image
        Predicted rooms.
    
    """
    if not os.path.exists(os.path.join(os.getcwd(), outdir + '/' + str(pltiter) + '/')):
        os.mkdir(os.path.join(os.getcwd(), outdir + '/' + str(pltiter) + '/'))
    im = img[0].numpy()
    imsave(os.path.join(os.getcwd(), outdir + '/' + str(pltiter) + '/' + name + '_image.png'), im)
    lcw = convert_one_hot_to_image(logcw)[0].numpy()
    imsave(os.path.join(os.getcwd(), outdir + '/' + str(pltiter) + '/' + name + '_close_walls.png'), img_as_float(lcw))
    ents = identify_bounds(lcw)
    imsave(os.path.join(os.getcwd(), outdir + '/' + str(pltiter) + '/' + name + '_doors_windows.png'), ents)
    lr = convert_one_hot_to_image(logr, dtype='int')[0].numpy().astype(np.uint8)
    imsave(os.path.join(os.getcwd(), outdir + '/' + str(pltiter) + '/' + name + '_rooms_pred.png'), lr)
    figure = plt.figure()
    ax1 = plt.subplot(2,3,1);plt.imshow(img[0].numpy());plt.xticks([]);plt.yticks([]);plt.grid(False)
    ax4 = plt.subplot(2,3,5);plt.imshow(convert_one_hot_to_image(logcw)[0].numpy());plt.xticks([]);plt.yticks([]);plt.grid(False)
    ax5 = plt.subplot(2,3,6);plt.imshow(convert_one_hot_to_image(logr)[0].numpy());plt.xticks([]);plt.yticks([]);plt.grid(False)
    figure.savefig(outdir + '/' + str(pltiter) + '_combined.png')
    plt.close()
    return figure


def infer(config):
    """Main run function.

    Parameters
    ----------
    config : dict or argparse namespace
        Dictionary of {'batchsize':2, 'lr':1e-4, 'wd':1e-5, 
                       'epochs':1000, 'logdir':'./log/store', 
                       'saveTensorInterval':10, 'saveModelInterval':2, 
                       'restore':'./log/store/', 'outdir':'./out', train=True}

    Returns
    -------
    None.

    """
    # initialization
    logdir=config['logdir']
    writer = tf.summary.create_file_writer(logdir) 
    pltiter = 0
    dataset,model,optim = init(config)
    if config['outdir'] is not None:
        if not os.path.exists(config['outdir']):
            os.mkdir(config['outdir'])
    if config['checkpoint'] is not None:
        print("Loading weights from {}".format(config['checkpoint']))
        latest = tf.train.latest_checkpoint(config['checkpoint'])
        model.load_weights(latest)
    for data in list(dataset.shuffle(400).batch(config['batchsize'])):
        img = decodeRawInfer(data)
        img = preprocessInfer(img)
        logits_r,logits_cw = model(img,training=False)
        
        name = data['zname']
        name = str(name.numpy()[0]).split("'")[-2]
        f = infer_image_grid(img, logits_r,logits_cw, pltiter, name, config['outdir'])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--batchsize',type=int,default=1)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--logdir',type=str,default='log/store')
    p.add_argument('--checkpoint',type=str,default=None)
    p.add_argument('--outdir',type=str,default='./out')
    p.add_argument('--train',type=bool,default=False)
    args = p.parse_args()
    infer(args)




