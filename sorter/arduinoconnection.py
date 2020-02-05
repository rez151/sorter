import serial, time
from IPython.core.magics import logging
from pynput.keyboard import Key, Listener


class ArduinoConnection:

    def __init__(self, port, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.arduino = serial.Serial(self.port, self.baudrate)
        time.sleep(1)  # give the connection a second to settle

    def __del__(self):
        del self.arduino

    def tilt_right(self):
        self.arduino.write(b'r')

    def tilt_left(self):
        self.arduino.write(b'l')

    def center(self):
        self.arduino.write(b'm')

    def shake(self):
        self.arduino.write(b'w')


"""
def on_press(key):
    a = key

    if a == Key.right:
        arduino.write(b'r')

    if a == Key.left:
        arduino.write(b'l')

    if a == Key.up:
        arduino.write(b'm')

    if a == Key.down:
        arduino.write(b'w')

    if a == Key.backspace:
        arduino.close()


with Listener(on_press=on_press) as listener:
    listener.join()
"""
