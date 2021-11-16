import tensorflow as tf
import argparse

def converter(config):
    model=tf.keras.models.load_model(config.modeldir)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_new_converter = True
    tflite_model = converter.convert()
    with open(config.tflitedir,'wb') as f:
        f.write(tflite_model)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--modeldir',type=str,default='model/store')
    p.add_argument('--tflitedir',type=str,default='model/store/model.tflite')
    args = p.parse_args()
    converter(args)

