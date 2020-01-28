import numpy as np
import tensorflow as tf
# from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import ImageDataGenerator
from tensorflow_core.python.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow_core.python.keras.layers.core import Dense
from tensorflow_core.python.keras.layers.pooling import GlobalAveragePooling2D
from tensorflow_core.python.keras.models import Model

# from tensorflow_core.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow_core.python.keras.optimizer_v2.adam import Adam

mobile = tf.keras.applications.mobilenet_v2.MobileNetV2()


def prepare_image(file):
    img_path = ''
    img = tf.keras.preprocessing.image.load_img(img_path + file, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


IMG_HEIGHT = 224
IMG_WIDTH = 224
base_model = MobileNetV2(weights='imagenet',
                         include_top=False, input_shape=(
        IMG_HEIGHT, IMG_WIDTH, 3))  # imports the mobilenet model and discards the last 1000 neuron layer.

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(
    x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
x = Dense(1024, activation='relu')(x)  # dense layer 2
x = Dense(512, activation='relu')(x)  # dense layer 3
preds = Dense(3, activation='softmax')(x)  # final layer with softmax activation

model = Model(inputs=base_model.input, outputs=preds)
# specify the inputs
# specify the outputs
# now a model has been created based on our architecture


for i, layer in enumerate(model.layers):
    print(i, layer.name)

# for layer in model.layers:
#    layer.trainable = False
# or if we want to set the first 20 layers of the network to be non-trainable
for layer in model.layers[:154]:
    layer.trainable = False
for layer in model.layers[154:]:
    layer.trainable = True

PATH = '/home/resi/PycharmProjects/sorter/sorter/datatrain'

batch_size = 64
epochs = 50

learning_rate = 0.001

train_image_generator = ImageDataGenerator(zoom_range=0.2,
                                           shear_range=0.2,
                                           horizontal_flip=True,
                                           validation_split=0.15,
                                           preprocessing_function=preprocess_input
                                           )

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=PATH,
                                                           shuffle=False,
                                                           color_mode='rgb',
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical',
                                                           subset='training'
                                                           )

validation_generator = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                                 directory=PATH,
                                                                 shuffle=False,
                                                                 color_mode='rgb',
                                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                 class_mode='categorical',
                                                                 subset='validation'
                                                                 )
adam = Adam(learning_rate=0.0005)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

step_size_train = train_data_gen.n // train_data_gen.batch_size
model.fit(x=train_data_gen,
          steps_per_epoch=step_size_train,
          epochs=epochs,
          validation_data=validation_generator,
          validation_steps=validation_generator.n // validation_generator.batch_size,
          )
