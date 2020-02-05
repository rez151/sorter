#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# Support module generated by PAGE version 4.26
#  in conjunction with Tcl version 8.6
#    Jan 27, 2020 03:45:50 PM CET  platform: Linux
#    Feb 04, 2020 04:48:16 PM CET  platform: Linux

import sys
from TinySorter1 import vid

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
    global combobox
    combobox = tk.StringVar()
    global spinbox
    spinbox = tk.StringVar()
    global learningRate
    learningRate = tk.StringVar()


def takePic1(p1):
    ret, frame = vid.get_frame()
    print('TinySorter1_support.takePic1')
    sys.stdout.flush()


def takePic2(p1):
    ret, frame = vid.get_frame()
    print('TinySorter1_support.takePic2')
    sys.stdout.flush()


def takePic3(p1):
    ret, frame = vid.get_frame()
    print('TinySorter1_support.takePic3')
    sys.stdout.flush()


def exportModel(p1):
    print('TinySorter1_support.exportModel')
    sys.stdout.flush()


def reset(p1):
    print('TinySorter1_support.reset')
    sys.stdout.flush()


def startSorting(p1):
    print('TinySorter1_support.startSorting')
    sys.stdout.flush()


def trainModel(p1):
    print('TinySorter1_support.trainModel')
    sys.stdout.flush()


def init(top, gui, *args, **kwargs):
    global w, top_level, root
    w = gui
    top_level = top
    root = top


def destroy_window():
    # Function which closes the window.
    global top_level
    top_level.destroy()
    top_level = None


if __name__ == '__main__':
    import TinySorter1

    TinySorter1.vp_start_gui()