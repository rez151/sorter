import numpy as np
import tensorflow as tf
from PIL import Image
from keras_applications.mobilenet_v2 import preprocess_input
from tensorflow_core.python.keras.applications.mobilenet_v2 import decode_predictions

model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', classes=2)

img_path = '/home/resi/PycharmProjects/sorter/sorter/weihnachtsmuetze/alu0.png'

im = Image.open(img_path)
print(type(im))
im = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))  # -> PIL image
doc = tf.keras.preprocessing.image.img_to_array(im)  # -> numpy array
doc = np.expand_dims(doc, axis=0)

# x = preprocess_input(x)

preds = model.predict(doc)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
