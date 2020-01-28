import glob
import os
import cv2
import numpy as np
import tensorflow as tf

from sorter.transferclassifier import TransferClassifier


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


def delete_dataset():
    files1 = glob.glob('/home/resi/PycharmProjects/sorter/sorter/datatrain/empty/*')
    files2 = glob.glob('/home/resi/PycharmProjects/sorter/sorter/datatrain/stressball/*')
    files3 = glob.glob('/home/resi/PycharmProjects/sorter/sorter/datatrain/weihnachtsmuetze/*')

    for f in files1:
        os.remove(f)
    for f in files2:
        os.remove(f)
    for f in files3:
        os.remove(f)


def run():
    cap = cv2.VideoCapture(2)
    counter_class1 = 0
    counter_class2 = 0
    counter_class3 = 0

    class1_path = "datatrain/weihnachtsmuetze"
    class2_path = "datatrain/stressball"
    class3_path = "datatrain/empty"

    create_folder(class1_path)
    create_folder(class2_path)
    create_folder(class3_path)

    transfer_classifier = TransferClassifier()

    while True:
        ret, frame = cap.read()
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
            transfer_classifier.load_data()

            print("start training")
            accuracy = transfer_classifier.train()
            print("Model Accuracy:", accuracy)

            transfer_classifier.saveModel()
            continue
        if k == ord('p'):
            print("predict")

            cv2.imwrite('/home/resi/PycharmProjects/sorter/sorter/predict/predimg.png', frame)
            frame = preprocess_sample('/home/resi/PycharmProjects/sorter/sorter/predict/predimg.png')
            prediction = transfer_classifier.predict_external(frame)

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


def main():
    run()


if __name__ == '__main__':
    main()
