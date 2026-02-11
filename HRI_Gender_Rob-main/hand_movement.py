# hand_simple.py

import time
import board
import busio
from adafruit_pca9685 import PCA9685

I2C_ADDRESS = 0x40
FREQ_HZ = 50

CHANNELS = {
    "thumb": 0,
    "index": 1,
    "middle": 2,
    "ring": 3,
    "pinky": 4,
}

# 👇 כאן את מכיילת פעם אחת וזהו
POSES = {
    "thumb":  {"open": 40, "closed": 120},
    "index":  {"open": 30, "closed": 140},
    "middle": {"open": 35, "closed": 145},
    "ring":   {"open": 20, "closed": 150},
    "pinky":  {"open": 10, "closed": 160},
}

MIN_US = 500
MAX_US = 2500

def deg_to_duty(deg):
    pulse_us = MIN_US + (MAX_US - MIN_US) * (deg / 180.0)
    period_us = 1_000_000 / FREQ_HZ
    return int((pulse_us / period_us) * 65535)

class Hand:
    def __init__(self):
        i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(i2c, address=I2C_ADDRESS)
        self.pca.frequency = FREQ_HZ

    def set_finger(self, finger, state):
        deg = POSES[finger][state]
        self.pca.channels[CHANNELS[finger]].duty_cycle = deg_to_duty(deg)

    def fist(self):
        for f in CHANNELS:
            self.set_finger(f, "closed")

    def open_hand(self):
        for f in CHANNELS:
            self.set_finger(f, "open")

    def thumb_up(self):
        self.fist()
        self.set_finger("thumb", "open")

    def pinky_up(self):
        self.fist()
        self.set_finger("pinky", "open")

hand = Hand()

# Startup animation
hand.open_hand()
time.sleep(0.7)
hand.fist()
time.sleep(0.7)

def on_gender(label: str):
    """
    label: "male" | "female" | "neutral" (or None/anything else -> neutral)
    """
    label = (label or "").strip().lower()
    if label == "male":
        hand.thumb_up()
        print("Gesture: MALE -> thumb up")
    elif label == "female":
        hand.pinky_up()
        print("Gesture: FEMALE -> pinky up")
    else:
        hand.fist()
        print("Gesture: NEUTRAL -> fist")

# Optional: quick manual CLI test
if __name__ == "__main__":
    print("Commands: m=male, f=female, n=neutral, q=quit")
    try:
        while True:
            cmd = input("> ").strip().lower()
            if cmd == "q":
                break
            elif cmd == "m":
                on_gender("male")
            elif cmd == "f":
                on_gender("female")
            else:
                on_gender("neutral")
    finally:
        hand.deinit()