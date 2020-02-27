import serial
import time


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
