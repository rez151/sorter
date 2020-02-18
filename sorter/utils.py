import glob
import os
import tensorflow as tf
import numpy as np


def create_folder(name):
    path = "./" + name

    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % name)
        else:
            print("Successfully created the directory %s " % name)
    else:
        print("Directory already exists")


def preprocess_sample(sample):
    im = tf.keras.preprocessing.image.load_img(sample, target_size=(224, 224))  # -> PIL image
    doc = tf.keras.preprocessing.image.img_to_array(im)  # -> numpy array
    doc = np.expand_dims(doc, axis=0)
    return doc
