import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow_core.python.keras.optimizer_v2.adamax import Adamax
from tensorflow_core.python.keras.optimizers import Adam

PATH = "/home/resi/PycharmProjects/sorter/sorter/datatrain"

batch_size = 16
epochs = 5
IMG_HEIGHT = 224
IMG_WIDTH = 224
learning_rate = 0.001

train_image_generator = ImageDataGenerator(rescale=1. / 255,
                                           zoom_range=0.2,
                                           shear_range=0.2,
                                           horizontal_flip=True,
                                           validation_split=0.15)

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=PATH,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical',
                                                           subset='training'
                                                           )

validation_generator = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                                 directory=PATH,
                                                                 shuffle=True,
                                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                 class_mode='categorical',
                                                                 subset='validation'
                                                                 )

model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights=None, classes=3)

model.compile(optimizer="nadam",
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=200 // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=64 // batch_size
)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

print(acc, val_acc, loss, val_loss)
