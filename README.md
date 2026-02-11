HRI Gender-Based Robotic Hand System
Project Description

This project implements a Human–Robot Interaction (HRI) system using:

Raspberry Pi

USB Camera

TensorFlow Lite gender classification

OpenCV face detection

PCA9685 I²C servo controller

5-servo robotic hand

The system detects a face, classifies gender, and performs a physical gesture using a robotic hand:

Male → Thumb Up

Female → Pinky Up

No face detected → Fist

When the system starts, the hand performs an open → close animation to indicate successful initialization.

System Architecture

Camera
↓
OpenCV (Face Detection)
↓
TensorFlow Lite (Gender Classification)
↓
Main Controller Logic
↓
I²C Servo Controller (PCA9685)
↓
Robotic Hand

Hardware Requirements

Raspberry Pi (I²C enabled)

USB Camera

PCA9685 Servo Controller (I²C)

5 Servo Motors (robotic hand)

External 5–6V power supply for servos

Shared ground between Raspberry Pi and servo power supply

Raspberry Pi I²C Wiring
Raspberry Pi Pin	Function
Pin 3	SDA
Pin 5	SCL
Pin 6	GND
Pin 2 or 4	5V
Software Requirements

Python 3.11

OpenCV

NumPy

TensorFlow Lite (tflite-runtime)

Adafruit Blinka

Adafruit PCA9685 library

Installation
1. Install system packages
sudo apt update
sudo apt install -y python3-opencv python3-smbus i2c-tools python3-venv python3-pip

2. Create virtual environment
python3 -m venv --system-site-packages ~/inmoov-venv
source ~/inmoov-venv/bin/activate

3. Install Python dependencies
python -m pip install --upgrade pip
python -m pip install numpy adafruit-blinka adafruit-circuitpython-pca9685
python -m pip install --extra-index-url https://www.piwheels.org/simple tflite-runtime

Model Files

Create a folder named models and place:

models/
 ├── haarcascade_frontalface_default.xml
 └── GenderClass_06_03-20-08.tflite

Running the System

Activate the virtual environment:

source ~/inmoov-venv/bin/activate


Run:

python main.py


Press q to exit.

Gesture Logic
Condition	Gesture
Startup	Open → Close
Male	Thumb Up
Female	Pinky Up
No Face	Fist

Finger open/closed angles are manually defined in the hand control file.

Troubleshooting
I²C Not Detected
sudo i2cdetect -y 1


Expected address: 40

If empty:

Check SDA/SCL wiring

Enable I²C via raspi-config

Verify shared ground

No module named "board"

Ensure:

Virtual environment is activated

adafruit-blinka is installed

Script is not run using system Python

No module named "cv2"

Install:

sudo apt install python3-opencv


Recreate virtual environment with --system-site-packages.

Ethical Considerations

The system performs visual gender classification for demonstration purposes only.
No images or personal data are stored.
