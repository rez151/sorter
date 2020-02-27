import os
import shutil
import threading
import time

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


# initialisierung GUI-Variablen
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


# onButtonClick Funktion für Klasse 1 Bild aufnehmen
def takePic1(p1):
    global class1_name, counter_class1
    class1_name = w.Class1.get("1.0", 'end-1c')  # Klassenname aus Textfeld lesen
    w.Class1.configure(state=tk.DISABLED)  # Eingabefeld darf nichtmehr verändert werden
    w.Class1.configure(background='#0071C5')

    # Erstelle Ordner für die Klasse
    classpath = dataset_path + class1_name
    if not os.path.exists(classpath):
        create_folder(classpath)

    # Speicher aktuelles Bild in den erstellten Ordner
    filepath = classpath + "/" + str(counter_class1.get()) + ".png"
    vid.save_frame(filepath)

    counter_class1.set(counter_class1.get() + 1)


# Analog zu takePic1
def takePic2(p1):
    global class2_name, counter_class2
    class2_name = w.Class2.get("1.0", 'end-1c')
    w.Class2.configure(state=tk.DISABLED)
    w.Class2.configure(background='#0071C5')

    classpath = dataset_path + class2_name
    if not os.path.exists(classpath):
        create_folder(classpath)

    filepath = classpath + "/" + str(counter_class2.get()) + ".png"
    vid.save_frame(filepath)

    counter_class2.set(counter_class2.get() + 1)


# Analog zu takePic1
def takePic3(p1):
    global class3_name, counter_class3
    # class3_name = "leer"
    w.Class3.configure(background='#0071C5')

    classpath = dataset_path + class3_name
    if not os.path.exists(classpath):
        create_folder(classpath)

    filepath = classpath + "/" + str(counter_class3.get()) + ".png"
    vid.save_frame(filepath)

    counter_class3.set(counter_class3.get() + 1)


# onButtonClick für Schütteln on/off
def shake(p1):
    global shaking

    if not shaking:
        arduino.shake()
        shaking = True
    else:
        arduino.center()
        shaking = False


# onButtonClick für Zurücksetzen
def reset(p1):
    global counter_class1, counter_class2, counter_class3
    for x in os.listdir(dataset_path):  #
        shutil.rmtree(dataset_path + x)  # lösche Datensatz

    # counter zurücksetzen
    counter_class1.set(0)
    counter_class2.set(0)
    counter_class3.set(0)

    # Textfelder wieder bearbeitbar machen
    w.Class1.configure(state=tk.NORMAL)
    w.Class1.configure(background='white')
    w.Class2.configure(state=tk.NORMAL)
    w.Class2.configure(background='white')
    w.Class3.configure(state=tk.NORMAL)
    w.Class3.configure(background='white')

    # transfer_classifier löschen
    global transfer_classifier
    del transfer_classifier

    # infos zurücksetzen
    global info1, info2
    info1.set("noch nichts gelernt")
    info2.set("Sortierer nicht bereit")


# K nur wählbar zwischen 1 und 100
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


# wird in eigenem Tread gestartet um die GUI Cam nicht einzufrieren
def sorting_loop():
    global sorting

    while sorting:

        vid.save_frame("./sorter/predict/predimg.png")  # aktuelles Bild speichern
        frame = preprocess_sample('./sorter/predict/predimg.png')  # gespeichertes bild laden und preprocessen
        # frame = preprocess_sample(frame) # direkt preprocessen ohne zwischenspeichern hat nicht geklappt

        prediction = transfer_classifier.predict_external(frame)

        print("Prediction: " + str(prediction))
        info2.set(str(prediction[0]))  # prediction in der GUI ausgeben

        # braucht kleinen sleep, sonst fährt er zu früh wieder zurück
        # mit den sleep werten muss unter umständen bisschen rumgespielt werden
        if prediction[0] == class1_name:
            arduino.tilt_left()
            time.sleep(0.5)
            arduino.center()
        if prediction[0] == class2_name:
            arduino.tilt_right()
            time.sleep(0.5)
            arduino.center()
        if prediction[0] == class3_name:
            # arduino.shake() # falls es bei leer prediction shaken soll
            time.sleep(0.5)
            arduino.center()


# onButtonClick für Starte Sortieren
def startSorting(p1):
    global info2
    global sorting

    if transfer_classifier is None:
        print("you have to train a classifier first")
        return
    sorting = True
    # Starte in neuem Thread, da sonst die GUI Webcam einfriert
    threading.Thread(target=sorting_loop).start()


# onButtonClick für Stoppe Sortieren
def stopSorting(p1):
    global info2
    info2.set("Sortierer gestoppt")
    global sorting
    sorting = False
    arduino.center()


# onButtonClick für lernen
def trainModel(p1):
    global k
    global info1, info2
    global counter_class1, counter_class2, counter_class3

    # prüfe ob alle Klassen Bilder haben
    if int(counter_class1.get()) == 0 or int(counter_class2.get()) == 0 or int(counter_class3.get()) == 0:
        info1.set("Zu wenig Bilder")
        return

    # K darf nicht größer sein als die Anzahl der Bilder im Datensatz - 1
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
arduino = ArduinoConnection('/dev/ttyACM1')  # Port kann Variieren!!! kann in der Arduino IDE nachgeschaut werden

# class1_name = "class1"
# class2_name = "class2"
class3_name = "leer"  # class1 und 2 werden aus den Textfeldern entnommen

dataset_path = "./sorter/datatrain/"

# global booleans zum starten und stoppen von sortieren und schütteln
global sorting, shaking
sorting = False
shaking = False

if __name__ == '__main__':
    import TinySorter1

    # Datensatz löschen beim neustart
    for x in os.listdir(dataset_path):
        shutil.rmtree(dataset_path + x)

    TinySorter1.vp_start_gui()
