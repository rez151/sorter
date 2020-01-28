import glob
import os
import pathlib
import cv2
import numpy as np
import tensorflow as tf

from PIL import Image
from joblib import dump, load
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

cap = cv2.VideoCapture(0)
counter_class1 = 0
counter_class2 = 0
counter_class3 = 0

class1_path = "datatrain/weihnachtsmuetze"
class2_path = "datatrain/stressball"
class3_path = "datatrain/empty"


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


create_folder(class1_path)
create_folder(class2_path)
create_folder(class3_path)


class BasicKNNClassifier:

    @staticmethod
    def preprocess_sample(sample):
        im = Image.open(sample)
        im = tf.keras.preprocessing.image.load_img(sample, target_size=(224, 224))  # -> PIL image
        doc = tf.keras.preprocessing.image.img_to_array(im)  # -> numpy array
        doc = np.expand_dims(doc, axis=0)
        return doc

    def load_data(self):
        data_dir = "/home/resi/PycharmProjects/sorter/sorter/datatrain/"
        data_dir = pathlib.Path(data_dir)

        CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])

        empty = list(data_dir.glob('empty/*'))
        ball = list(data_dir.glob('stressball/*'))
        muetze = list(data_dir.glob('weihnachtsmuetze/*'))

        print(len(empty))
        print(len(ball))
        print(len(muetze))

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
        data_train, data_test, target_train, target_test = train_test_split(self.datanp, self.targetnp,
                                                                            test_size=0.3,
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


basic_knn_classifier = BasicKNNClassifier()
basic_knn_classifier.loadModel()


def preprocess_sample(sample):
    # im = Image.fromarray(sample)

    # im = Image.open(sample)
    im = tf.keras.preprocessing.image.load_img(sample, target_size=(224, 224))  # -> PIL image
    doc = tf.keras.preprocessing.image.img_to_array(im)  # -> numpy array
    doc = np.expand_dims(doc, axis=0)
    return doc


def delete_dataset():
    # os.remove("/home/resi/PycharmProjects/sorter/sorter/datatrain/empty/*")
    # os.remove("/home/resi/PycharmProjects/sorter/sorter/datatrain/stressball/*")
    # os.remove("/home/resi/PycharmProjects/sorter/sorter/datatrain/weihnachtsmuetze/*")

    files1 = glob.glob('/home/resi/PycharmProjects/sorter/sorter/datatrain/empty/*')
    files2 = glob.glob('/home/resi/PycharmProjects/sorter/sorter/datatrain/stressball/*')
    files3 = glob.glob('/home/resi/PycharmProjects/sorter/sorter/datatrain/weihnachtsmuetze/*')

    for f in files1:
        os.remove(f)
    for f in files2:
        os.remove(f)
    for f in files3:
        os.remove(f)


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frameq
    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == ord('1'):
        cv2.imwrite(class1_path + "/weihnachtsmueze" + str(counter_class1) + '.png', frame)
        print("image added to class class 1")
        counter_class1 += 1
        continue
    if k == ord('2'):
        cv2.imwrite(class2_path + '/stressball' + str(counter_class2) + '.png', frame)
        print("image added to class class 2")
        counter_class2 += 1
        continue
    if k == ord('3'):
        cv2.imwrite(class3_path + '/empty' + str(counter_class3) + '.png', frame)
        print("image added to class class 3")
        counter_class3 += 1
        continue
    if k == ord('l'):
        print("loading dataset")
        basic_knn_classifier.load_data()
        print("done")
        print("start training")
        accuracy = basic_knn_classifier.train()
        print("Model Accuracy:", accuracy)
        basic_knn_classifier.saveModel()

        continue

    if k == ord('p'):
        print("predict")
        cv2.imwrite('/home/resi/PycharmProjects/sorter/sorter/predict/predimg.png', frame)

        frame = preprocess_sample('/home/resi/PycharmProjects/sorter/sorter/predict/predimg.png')
        frame = np.reshape(frame, (1, 224 * 224 * 3))

        prediction = basic_knn_classifier.predict(frame)
        print("Prediction for {0} => \n{1}".format(frame, prediction))

        continue

    if k == ord("r"):
        delete_dataset()
        continue
    if k == ord('q'):
        print("q")
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()