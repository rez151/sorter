import os
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
    class3_name = w.Class3.get("1.0", 'end-1c')
    w.Class3.configure(state=tk.DISABLED)
    w.Class3.configure(background='#0071C5')

    if not os.path.exists(dataset_path + class3_name):
        create_folder(dataset_path + class3_name)

    filepath = dataset_path + class3_name + "/" + str(counter_class3.get()) + ".png"
    vid.save_frame(filepath)

    counter_class3.set(counter_class3.get() + 1)


def exportModel(p1):
    pass


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


def startSorting(p1):
    print('TinySorter1_support.startSorting')
    sys.stdout.flush()

    if transfer_classifier is None:
        print("you have to train a classifier first")
        return

    print("predict")

    vid.save_frame("./sorter/predict/predimg.png")
    frame = preprocess_sample('./sorter/predict/predimg.png')
    # frame = preprocess_sample(frame)
    prediction = transfer_classifier.predict_external(frame)

    print("Prediction: " + str(prediction))

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
    pass


def trainModel(p1):
    global transfer_classifier
    transfer_classifier = TransferClassifier(class1_name, class2_name, class3_name, k.get())

    print("loading dataset")
    transfer_classifier.load_data()

    print("start training")
    accuracy = transfer_classifier.train()
    print("Model Accuracy:", accuracy)

    transfer_classifier.save_model()


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
arduino = ArduinoConnection('/dev/ttyACM0')

class1_name = "class1"
class2_name = "class2"
class3_name = "empty"

dataset_path = "./sorter/datatrain/"

if __name__ == '__main__':
    import TinySorter1

    # TinySorter1

    TinySorter1.vp_start_gui()
