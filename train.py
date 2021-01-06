import tensorflow as tf
import io
import tqdm
from net import *
from loss import *
from data import *
import argparse
import os
import pandas as pd
from PIL import Image
from datetime import datetime
from skimage.io import imread, imsave
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
    dataset = loadDataset()
    #optim = tf.keras.optimizers.AdamW(learning_rate=config.lr,weight_decay=config.wd)
    optim = tf.keras.optimizers.Adam(learning_rate=config['lr'])
    return dataset,model,optim

def plot_to_image(figure, pltiter, directory, save=False):
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
        imSave.save(os.path.join(directory, str(pltiter) + '.png'))
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

def image_grid(img,bound,room,logr,logcw, pltiter, outdir):
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
    figure = plt.figure()
    ax1 = plt.subplot(2,3,1);plt.imshow(img[0].numpy());plt.xticks([]);plt.yticks([]);plt.grid(False)
    extent1 = ax1.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
    figure.savefig(outdir + '/' + str(pltiter) + '_image.png', bbox_inches=extent1)
    ax2 = plt.subplot(2,3,2);plt.imshow(bound[0].numpy());plt.xticks([]);plt.yticks([]);plt.grid(False)
    extent2 = ax2.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
    figure.savefig(outdir + '/' + str(pltiter) + '_bounds.png', bbox_inches=extent2)
    ax3 = plt.subplot(2,3,3);plt.imshow(room[0].numpy());plt.xticks([]);plt.yticks([]);plt.grid(False)
    extent3 = ax3.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
    figure.savefig(outdir + '/' + str(pltiter) + '_rooms.png', bbox_inches=extent3)
    ax4 = plt.subplot(2,3,5);plt.imshow(convert_one_hot_to_image(logcw)[0].numpy());plt.xticks([]);plt.yticks([]);plt.grid(False)
    extent4 = ax4.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
    figure.savefig(outdir + '/' + str(pltiter) + '_logcw.png', bbox_inches=extent4)
    ax5 = plt.subplot(2,3,6);plt.imshow(convert_one_hot_to_image(logr)[0].numpy());plt.xticks([]);plt.yticks([]);plt.grid(False)
    extent5 = ax4.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
    figure.savefig(outdir + '/' + str(pltiter) + '_logr.png', bbox_inches=extent5)
    return figure

def image_single(img, bound, room, logr, logcw):
    im = plt.imshow(img[0].numpy())
    imsave('image.png', im)
  #  imsave('bound.png', bound[0].numpy())
  #  imsave('room.png', room[0].numpy())
  #  imsave('roomTypes.png', logr[0].numpy())
  #  imsave('entries.png', logcw[0].numpy())

def main(config):
    """Main run function.

    Parameters
    ----------
    config : dict or argparse namespace
        Dictionary of {'batchsize':2, 'lr':1e-4, 'wd':1e-5, 
                       'epochs':1000, 'logdir':'./log/store', 
                       'saveTensorInterval':10, 'saveModelInterval':2, 
                       'restore':'./log/store/', 'outdir':'./out'}

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
    losses1=[]
    losses2=[]
    totalLosses=[]
    #load weights from previous training if re-starting
    if config['restore'] is not None:
        print("Loading weights from {}".format(config['restore']))
        latest = tf.train.latest_checkpoint(config['restore'])
        model.load_weights(latest)

    # training loop
    for epoch in range(config['epochs']):
        for data in list(dataset.shuffle(400).batch(config['batchsize'])):
            # forward
            img,bound,room = decodeAllRaw(data)
            img,bound,room,hb,hr = preprocess(img,bound,room)
            with tf.GradientTape() as tape:
                logits_r,logits_cw = model(img,training=True)
                loss1 = balanced_entropy(logits_r,hr)
                loss2 = balanced_entropy(logits_cw,hb)
                w1,w2 = cross_two_tasks_weight(hr,hb)
                loss = w1*loss1+w2*loss2
            # backward
            grads = tape.gradient(loss,model.trainable_weights)
            optim.apply_gradients(zip(grads,model.trainable_weights))
        
            # plot progress
            if config['outdir'] is not None:
                if pltiter%config['saveTensorInterval'] == 0:
                    f = image_grid(img,bound,room,logits_r,logits_cw, pltiter, config['outdir'])
                    im = plot_to_image(f, pltiter, config['outdir'], save=True)
           #         image_single(img, bound,room,logits_r,logits_cw)
    
                    with writer.as_default():
                        tf.summary.scalar("Loss",loss.numpy(),step=pltiter)
                        tf.summary.scalar("LossR",loss1.numpy(),step=pltiter)
                        tf.summary.scalar("LossB",loss2.numpy(),step=pltiter)
                        tf.summary.image("Data",im,step=pltiter)
                    writer.flush()
            pltiter += 1

        # save model
        if epoch%config['saveModelInterval'] == 0:
            model.save_weights(logdir+'/G')
            tf.keras.callbacks.ModelCheckpoint(filepath=config['logdir'],
                                                 save_weights_only=False,
                                                 verbose=1)
            print('[INFO] Saving Model')
        print('[INFO] Epoch {}'.format(epoch) + ' loss ' + str(loss) + ' roomTypeLoss: '  + str(loss1) + ' roomBoundLoss' + str(loss2))
        losses1.append(loss1)
        losses2.append(loss2)
        totalLosses.append(loss)
        now = datetime.now()
        now = str(now).split(' ')[0]
        df = pd.DataFrame([loss1, loss2, totalLosses]).T
        df.columns=['Loss_RoomType', 'Loss_RoomBound', 'TotalLoss']
        df.to_csv(os.path.join(logdir, 'losses_' + str(now) + '.csv'))
  #  pdb.set_trace() #this is for debugging and is not needed

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--batchsize',type=int,default=2)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--wd',type=float,default=1e-5)
    p.add_argument('--epochs',type=int,default=1000)
    p.add_argument('--logdir',type=str,default='log/store')
    p.add_argument('--saveTensorInterval',type=int,default=10)
    p.add_argument('--saveModelInterval',type=int,default=20)
    p.add_argument('--restore',type=str,default=None)
    p.add_argument('--outdir',type=str,default='./out')
    args = p.parse_args()
    main(args)







