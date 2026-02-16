import time
import board
import busio
from adafruit_pca9685 import PCA9685

# ====== CONFIG ======
I2C_ADDRESS = 0x40   # הכי נפוץ ב-PCA9685
FREQ_HZ = 50         # servo PWM frequency
DELTA_DEG = 15

# 5 servos on channels 0..4 (שני אם הערוצים אצלך אחרים)
CHANNELS = [0, 1, 2, 3, 4]

# סרווים סטנדרטיים: ~500us עד ~2500us לרוב 0..180°
MIN_US = 500
MAX_US = 2500

# ====================

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def deg_to_duty_cycle(deg: float) -> int:
    deg = clamp(deg, 0.0, 180.0)
    pulse_us = MIN_US + (MAX_US - MIN_US) * (deg / 180.0)
    period_us = 1_000_000.0 / FREQ_HZ
    duty = int((pulse_us / period_us) * 65535)
    return clamp(duty, 0, 65535)

def set_servo(pca: PCA9685, ch: int, deg: float):
    pca.channels[ch].duty_cycle = deg_to_duty_cycle(deg)

def main():
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c, address=I2C_ADDRESS)
    pca.frequency = FREQ_HZ

    base = 90  # נקודת אמצע בטוחה לרוב הסרווים

    # למרכז
    for ch in CHANNELS:
        set_servo(pca, ch, base)
    time.sleep(1)

    # להזיז +15°
    for ch in CHANNELS:
        set_servo(pca, ch, base + DELTA_DEG)
        print(f"CH{ch}: {base} -> {base + DELTA_DEG}")
        time.sleep(0.7)

    # לחזור למרכז
    for ch in CHANNELS:
        set_servo(pca, ch, base)
    time.sleep(1)

    pca.deinit()
    print("done")

if __name__ == "__main__":
    main()
