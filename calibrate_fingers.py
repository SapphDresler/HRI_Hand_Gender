import time
import board
import busio
from adafruit_pca9685 import PCA9685

I2C_ADDRESS = 0x40
FREQ_HZ = 50

# channels for the 5 servos (change if needed)
FINGERS = [
    ("thumb", 0),
    ("index", 1),
    ("middle", 2),
    ("ring", 3),
    ("pinky", 4),
]

DELTA = 15

# Set True for fingers that are reversed
INVERT = {
    "thumb": False,
    "index": False,
    "middle": False,
    "ring": False,
    "pinky": False,
}

MIN_US = 500
MAX_US = 2500

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def deg_to_duty(deg: float) -> int:
    deg = clamp(deg, 0.0, 180.0)
    pulse_us = MIN_US + (MAX_US - MIN_US) * (deg / 180.0)
    period_us = 1_000_000.0 / FREQ_HZ
    duty = int((pulse_us / period_us) * 65535)
    return clamp(duty, 0, 65535)

def set_deg(pca: PCA9685, ch: int, deg: float):
    pca.channels[ch].duty_cycle = deg_to_duty(deg)

def main():
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c, address=I2C_ADDRESS)
    pca.frequency = FREQ_HZ

    # "Neutral" is whatever you choose to command now (we'll use 90 as a safe baseline),
    # but we apply the SAME baseline to all + direction normalization via INVERT.
    neutral = 90

    # move to neutral first (safe consistent start)
    for name, ch in FINGERS:
        set_deg(pca, ch, neutral)
    time.sleep(1)

    # apply normalized +15 (or -15 if inverted)
    for name, ch in FINGERS:
        step = -DELTA if INVERT[name] else DELTA
        set_deg(pca, ch, neutral + step)
        print(f"{name}: {'-' if INVERT[name] else '+'}{DELTA}")
        time.sleep(0.7)

    # back to neutral
    for name, ch in FINGERS:
        set_deg(pca, ch, neutral)
    time.sleep(1)

    pca.deinit()
    print("done")

if __name__ == "__main__":
    main()