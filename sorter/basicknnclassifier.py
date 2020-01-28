import cv2
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from joblib import dump, load
import os
import pathlib
import numpy as np
import IPython.display as display
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf


class BasicKNNClassifier:

    def preprocess_sample(self, sample):
        im = Image.open(sample)
        im = tf.keras.preprocessing.image.load_img(sample, target_size=(224, 224))  # -> PIL image
        doc = tf.keras.preprocessing.image.img_to_array(im)  # -> numpy array
        doc = np.expand_dims(doc, axis=0)
        return doc

    def load_data(self):
        data_dir = "datatrain"
        data_dir = pathlib.Path(data_dir)

        CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])

        empty = list(data_dir.glob('empty/*'))
        ball = list(data_dir.glob('ball/*'))
        muetze = list(data_dir.glob('muetze/*'))

        self.data = []
        self.target = []
        self.target_names = CLASS_NAMES

        print("arrays created")

        for sample in empty:
            self.data.append(self.preprocess_sample(sample))
            self.target.append(0)

        for sample in ball:
            self.data.append(self.preprocess_sample(sample))
            self.target.append(1)

        for sample in muetze:
            self.data.append(self.preprocess_sample(sample))
            self.target.append(2)

        self.datanp = np.array(self.data)
        size = len(self.datanp)
        self.datanp = np.reshape(self.datanp, (size, 224 * 224 * 3))
        self.targetnp = np.array(self.target)
        self.target_namesnp = np.array(self.target_names)

    def train(self):
        data_train, data_test, target_train, target_test = train_test_split(self.datanp, self.targetnp, test_size=0.3,
                                                                            random_state=12)

        self.classifier = KNeighborsClassifier()
        self.classifier.fit(data_train, target_train)

        target_pred = self.classifier.predict(data_test)
        accuracy = metrics.accuracy_score(target_test, target_pred)

        return accuracy

    def predict(self, external_input_sample):
        prediction_raw_values = self.classifier.predict(external_input_sample)
        prediction_resolved_values = [self.target_names[p] for p in prediction_raw_values]
        return prediction_resolved_values

    def saveModel(self):
        dump(self.classifier, 'trained_model.pkl')
        dump(self.target_names, 'trained_iris_model_targetNames.pkl')

    def loadModel(self):
        self.classifier = load('/home/resi/PycharmProjects/sorter/model/knn/trained_iris_model.pkl')
        self.target_names = load('/home/resi/PycharmProjects/sorter/model/knn/trained_iris_model_targetNames.pkl')
