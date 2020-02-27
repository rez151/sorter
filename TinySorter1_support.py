import os
import threading
import time

import cv2
import sys
import shutil

from sorter.arduinoconnection import ArduinoConnection
from sorter.transferclassifier import TransferClassifier
from sorter.utils import create_folder, preprocess_sample

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk

    py3 = False
except ImportError:
    import tkinter.ttk as ttk

    py3 = True


def set_Tk_var():
    global k
    k = tk.IntVar()
    k.set(1)

    global info1, info2
    info1 = tk.StringVar()
    info1.set("noch nichts gelernt")

    info2 = tk.StringVar()
    info2.set("Sortierer nicht bereit")

    global counter_class1, counter_class2, counter_class3
    counter_class1 = tk.IntVar()
    counter_class1.set(0)
    counter_class2 = tk.IntVar()
    counter_class2.set(0)
    counter_class3 = tk.IntVar()
    counter_class3.set(0)


def takePic1(p1):
    global class1_name, counter_class1
    class1_name = w.Class1.get("1.0", 'end-1c')
    w.Class1.configure(state=tk.DISABLED)
    w.Class1.configure(background='#0071C5')

    if not os.path.exists(dataset_path + class1_name):
        create_folder(dataset_path + class1_name)

    filepath = dataset_path + class1_name + "/" + str(counter_class1.get()) + ".png"
    vid.save_frame(filepath)

    counter_class1.set(counter_class1.get() + 1)


def takePic2(p1):
    global class2_name, counter_class2
    class2_name = w.Class2.get("1.0", 'end-1c')
    w.Class2.configure(state=tk.DISABLED)
    w.Class2.configure(background='#0071C5')

    if not os.path.exists(dataset_path + class2_name):
        create_folder(dataset_path + class2_name)

    filepath = dataset_path + class2_name + "/" + str(counter_class2.get()) + ".png"
    vid.save_frame(filepath)

    counter_class2.set(counter_class2.get() + 1)


def takePic3(p1):
    global class3_name, counter_class3
    # class3_name = "leer"
    w.Class3.configure(background='#0071C5')

    if not os.path.exists(dataset_path + class3_name):
        create_folder(dataset_path + class3_name)

    filepath = dataset_path + class3_name + "/" + str(counter_class3.get()) + ".png"
    vid.save_frame(filepath)

    counter_class3.set(counter_class3.get() + 1)


def shake(p1):
    global shaking

    if not shaking:
        arduino.shake()
        shaking = True
    else:
        arduino.center()
        shaking = False


def reset(p1):
    global counter_class1, counter_class2, counter_class3
    for x in os.listdir(dataset_path):
        shutil.rmtree(dataset_path + x)

    counter_class1.set(0)
    counter_class2.set(0)
    counter_class3.set(0)

    w.Class1.configure(state=tk.NORMAL)
    w.Class1.configure(background='white')
    w.Class2.configure(state=tk.NORMAL)
    w.Class2.configure(background='white')
    w.Class3.configure(state=tk.NORMAL)
    w.Class3.configure(background='white')

    global transfer_classifier
    del transfer_classifier

    global info1, info2
    info1.set("noch nichts gelernt")
    info2.set("Sortierer nicht bereit")


def increase_k(p1):
    global k

    tmp = k.get()
    if tmp < 99:
        k.set(tmp + 1)


def decrease_k(p1):
    global k

    tmp = k.get()
    if tmp > 1:
        k.set(tmp - 1)


def sorting_loop():
    global sorting

    while sorting:

        vid.save_frame("./sorter/predict/predimg.png")
        frame = preprocess_sample('./sorter/predict/predimg.png')
        # frame = preprocess_sample(frame)
        prediction = transfer_classifier.predict_external(frame)

        print("Prediction: " + str(prediction))
        info2.set(str(prediction[0]))

        if prediction[0] == class1_name:
            arduino.tilt_left()
            time.sleep(0.5)
            arduino.center()
        if prediction[0] == class2_name:
            arduino.tilt_right()
            time.sleep(0.5)
            arduino.center()
        if prediction[0] == class3_name:
            arduino.shake()
            time.sleep(0.5)
            arduino.center()


def startSorting(p1):
    global info2
    global sorting

    if transfer_classifier is None:
        print("you have to train a classifier first")
        return

    print("predict")

    sorting = True

    threading.Thread(target=sorting_loop).start()


def startSorting_old(p1):
    global info2
    global sorting
    sorting = True

    if transfer_classifier is None:
        print("you have to train a classifier first")
        return

    print("predict")

    vid.save_frame("./sorter/predict/predimg.png")
    frame = preprocess_sample('./sorter/predict/predimg.png')
    # frame = preprocess_sample(frame)
    prediction = transfer_classifier.predict_external(frame)

    print("Prediction: " + str(prediction))
    info2.set(str(prediction[0]))

    if prediction[0] == class1_name:
        arduino.tilt_left()
        time.sleep(1)
        arduino.center()
    if prediction[0] == class2_name:
        arduino.tilt_right()
        time.sleep(1)
        arduino.center()
    if prediction[0] == class3_name:
        arduino.shake()
        time.sleep(2)
        arduino.center()


def stopSorting(p1):
    global info2
    info2.set("Sortierer gestoppt")
    global sorting
    sorting = False
    arduino.center()


def trainModel(p1):
    global k
    global info1, info2
    global counter_class1, counter_class2, counter_class3

    if int(counter_class1.get()) == 0 or int(counter_class2.get()) == 0 or int(counter_class3.get()) == 0:
        info1.set("Zu wenig Bilder")
        return

    if k.get() >= int(counter_class1.get()) + int(counter_class2.get()) + int(counter_class3.get()) - 1:
        info1.set("K zu groß")
        return

    info1.set("KNN wird erstellt.")
    global transfer_classifier
    transfer_classifier = TransferClassifier(class1_name, class2_name, class3_name, k.get())

    info1.set("Datensatz wird geladen..")
    print("loading dataset")
    transfer_classifier.load_data()

    print("start training")
    info1.set("KNN lernt...")
    accuracy = transfer_classifier.train()
    print("Model Accuracy:", accuracy)
    info1.set("Genauigkeit: " + str("%3.2f" % accuracy))

    transfer_classifier.save_model()

    info2.set("Sortierer bereit")


def init(top, gui, vidsrc, *args, **kwargs):
    global w, top_level, root, vid
    w = gui
    top_level = top
    root = top
    vid = vidsrc


def destroy_window():
    # Function which closes the window.
    global top_level
    top_level.destroy()
    top_level = None


global transfer_classifier
arduino = ArduinoConnection('/dev/ttyACM1')

class1_name = "class1"
class2_name = "class2"
class3_name = "leer"

dataset_path = "./sorter/datatrain/"

global sorting, shaking
sorting = False
shaking = False

if __name__ == '__main__':
    import TinySorter1

    for x in os.listdir(dataset_path):
        shutil.rmtree(dataset_path + x)
    # TinySorter1

    TinySorter1.vp_start_gui()
