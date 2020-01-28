import serial, time
from IPython.core.magics import logging
from pynput.keyboard import Key, Listener

arduino = serial.Serial('/dev/ttyACM0', 115200)
time.sleep(1)  # give the connection a second to settle


def on_press(key):
    a = key

    if a == Key.right:
        arduino.write(b'r')

    if a == Key.left:
        arduino.write(b'l')

    if a == Key.up:
        arduino.write(b'm')

    if a == Key.down:
        arduino.close()


with Listener(on_press=on_press) as listener:
    listener.join()
