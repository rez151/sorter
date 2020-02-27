import os
import pathlib

import numpy as np
import tensorflow as tf
from PIL import Image
from joblib import load, dump
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tensorflow_core.python.keras.applications.mobilenet_v2 import MobileNetV2

# beide Links Simpel auf ein Plakat bringen!!
# https://github.com/google-coral/project-teachable  README!!!!
# https://observablehq.com/@makerslabemlyon/how-can-we-use-mobilenet-to-train-a-knn
class TransferClassifier:

    def __init__(self, class1_name, class2_name, class3_name, k=5):
        self.class1_name = class1_name
        self.class2_name = class2_name
        self.class3_name = class3_name
        self.k = k
        self.target = []
        self.data = []
        self.mobilenet = MobileNetV2(weights='imagenet', include_top=False)
        # if os.path.exists('trained_model.pkl') and os.path.exists('trained_model_targetNames.pkl'):
        #    self.knnclassifier = load('trained_model.pkl')
        #    self.target_names = load('trained_model_targetNames.pkl')

    @staticmethod
    def preprocess_sample(sample):
        im = Image.open(sample)
        im = tf.keras.preprocessing.image.load_img(sample, target_size=(224, 224))
        doc = tf.keras.preprocessing.image.img_to_array(im)  # -> numpy array
        doc = np.expand_dims(doc, axis=0)
        return doc

    def load_data(self):
        data_dir = "./sorter/datatrain/"
        data_dir = pathlib.Path(data_dir)

        # CLASS_NAMES = np.array([self.class1_name, self.class2_name, self.class3_name])
        # CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
        class_names = np.array([self.class1_name, self.class2_name, self.class3_name])
        class1 = list(data_dir.glob(self.class1_name + '/*'))
        class2 = list(data_dir.glob(self.class2_name + '/*'))
        class3 = list(data_dir.glob(self.class3_name + '/*'))

        print(len(class1))
        print(len(class2))
        print(len(class3))

        self.target_names = class_names

        print(class_names)

        for sample in class1:
            self.data.append(self.preprocess_sample(sample))
            self.target.append(1)

        for sample in class2:
            self.data.append(self.preprocess_sample(sample))
            self.target.append(2)

        for sample in class3:
            self.data.append(self.preprocess_sample(sample))
            self.target.append(3)

        self.datanp = np.array(self.data)
        size = len(self.datanp)
        self.targetnp = np.array(self.target)
        self.target_namesnp = np.array(self.target_names)

    def train(self):
        data_train, data_test, target_train, target_test = train_test_split(self.datanp, self.targetnp,
                                                                            test_size=0.3,
                                                                            random_state=12)

        evs_train = self.get_embedding_vectors(data_train)
        evs_test = self.get_embedding_vectors(data_test)
        self.knnclassifier = KNeighborsClassifier(self.k)
        self.knnclassifier.fit(evs_train, target_train)

        target_pred = self.knnclassifier.predict(evs_test)
        accuracy = metrics.accuracy_score(target_test, target_pred)

        return accuracy

    def predict_external(self, external_frame):
        external_frame = self.get_predicting_vectors(external_frame)
        prediction_raw_values = self.knnclassifier.predict(external_frame)
        prediction_resolved_values = [self.target_names[p - 1] for p in prediction_raw_values]
        return prediction_resolved_values

    def save_model(self):
        dump(self.knnclassifier, 'trained_model.pkl')
        dump(self.target_names, 'trained_model_targetNames.pkl')

    # def load_model(self):
    #    self.knnclassifier = load('trained_model.pkl')
    #    self.target_names = load('trained_model_targetNames.pkl')

    def get_predicting_vectors(self, data_train):
        embedding_vector = self.mobilenet.predict(data_train)

        embedding_vector = np.array(embedding_vector)
        embedding_vector = np.reshape(embedding_vector, (len(data_train), 1 * 7 * 7 * 1280))

        return embedding_vector

    def get_embedding_vectors(self, data_train):
        embedding_vectors = []
        for sample in data_train:
            embedding_vectors.append(self.mobilenet.predict(sample))

        embedding_vectors = np.array(embedding_vectors)
        embedding_vectors = np.reshape(embedding_vectors, (len(data_train), 1 * 7 * 7 * 1280))

        return embedding_vectors
