import tensorflow as tf
import io
import tqdm
from net import *
from loss import *
from data import *
import argparse
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def init(config):
    dataset = loadDataset()
    model = deepfloorplanModel()
    #optim = tf.keras.optimizers.AdamW(learning_rate=config.lr,weight_decay=config.wd)
    optim = tf.keras.optimizers.Adam(learning_rate=config.lr)
    return dataset,model,optim

def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

def image_grid(img,bound,room,logr,logcw):
    figure = plt.figure()
    plt.subplot(2,3,1);plt.imshow(img[0].numpy());plt.xticks([]);plt.yticks([]);plt.grid(False)
    plt.subplot(2,3,2);plt.imshow(bound[0].numpy());plt.xticks([]);plt.yticks([]);plt.grid(False)
    plt.subplot(2,3,3);plt.imshow(room[0].numpy());plt.xticks([]);plt.yticks([]);plt.grid(False)
    plt.subplot(2,3,5);plt.imshow(convert_one_hot_to_image(logcw)[0].numpy());plt.xticks([]);plt.yticks([]);plt.grid(False)
    plt.subplot(2,3,6);plt.imshow(convert_one_hot_to_image(logr)[0].numpy());plt.xticks([]);plt.yticks([]);plt.grid(False)
    return figure

def main(config):
    # initialization
    writer = tf.summary.create_file_writer(config.logdir); pltiter = 0
    dataset,model,optim = init(config)
    # training loop
    for epoch in range(config.epochs):
        print('[INFO] Epoch {}'.format(epoch))
        for data in list(dataset.shuffle(400).batch(config.batchsize)):
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
            if pltiter%config.saveTensorInterval == 0:
                f = image_grid(img,bound,room,logits_r,logits_cw)
                im = plot_to_image(f)
                with writer.as_default():
                    tf.summary.scalar("Loss",loss.numpy(),step=pltiter)
                    tf.summary.scalar("LossR",loss1.numpy(),step=pltiter)
                    tf.summary.scalar("LossB",loss2.numpy(),step=pltiter)
                    tf.summary.image("Data",im,step=pltiter)
                writer.flush()
            pltiter += 1

        # save model
        if epoch%config.saveModelInterval == 0:
            model.save_weights(config.logdir+'/G')
            model.save(config.modeldir)
            print('[INFO] Saving Model ...')

    pdb.set_trace()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--batchsize',type=int,default=2)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--wd',type=float,default=1e-5)
    p.add_argument('--epochs',type=int,default=1000)
    p.add_argument('--logdir',type=str,default='log/store')
    p.add_argument('--modeldir',type=str,default='model/store')
    p.add_argument('--saveTensorInterval',type=int,default=10)
    p.add_argument('--saveModelInterval',type=int,default=20)
    args = p.parse_args()
    main(args)







