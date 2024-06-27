import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
from picamera2 import Picamera2
from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import Servo

# Initialize the Picamera2
def initialize_camera():
    try:
        picam2 = Picamera2()
        picam2.preview_configuration.main.size = (1280, 720)
        picam2.preview_configuration.main.format = "RGB888"
        picam2.preview_configuration.align()
        picam2.configure("preview")
        return picam2
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return None


# Initialize current position
current_position_x = 0
current_position_y = 0
factory = PiGPIOFactory()
servo_x = Servo(18, min_pulse_width = 0.5/1000, max_pulse_width = 2.5/1000, pin_factory=factory)
servo_y = Servo(17, min_pulse_width = 0.5/1000, max_pulse_width = 2.5/1000, pin_factory=factory)


# Function to gradually change horizontal servo position
def move_servo_x(current_position_x, center_x):
    target_position = 0  # Center position
    step = 0.02  # Step size for servo movement
    if center_x < 640:
        current_position_x += step
        if current_position_x > 1:
            current_position_x = 1
    elif center_x > 640:
        current_position_x -= step
        if current_position_x < -1:
            current_position_x = -1
    else:
        return current_position_x  # Don't move if already centered
    servo_x.value = current_position_x
    return current_position_x

# Function to gradually change vertical servo position
def move_servo_y(current_position_y, center_y):
    target_position = 0  # Center position
    step = 0.02  # Step size for servo movement
    if center_y > 360:
        current_position_y += step
        if current_position_y > 1:
            current_position_y = 1
    elif center_y < 360:
        current_position_y -= step
        if current_position_y < -1:
            current_position_y = -1
    else:
        return current_position_y  # Don't move if already centered
    servo_y.value = current_position_y
    return current_position_y


class VideoStream:
    """Camera object that controls video streaming from the Picamera2"""
    def __init__(self, resolution=(640, 480), framerate=30):
        self.picam2 = initialize_camera()
        if self.picam2:
            try:
                self.picam2.stop()  # Ensure camera is stopped before configuring
                self.picam2.preview_configuration.main.size = resolution
                self.picam2.preview_configuration.main.format = "RGB888"
                self.picam2.preview_configuration.align()
                self.picam2.configure("preview")
                self.picam2.start()
                self.frame = None
                self.stopped = False
            except Exception as e:
                print(f"Error initializing camera in VideoStream: {e}")
                raise
        else:
            raise RuntimeError("Failed to initialize camera")

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.picam2.stop()
                return
            self.frame = self.picam2.capture_array()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.picam2.stop()

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in', required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite', default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt', default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects', default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.', default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection', action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

if use_TPU:
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

CWD_PATH = os.getcwd()

PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

if labels[0] == '???':
    del(labels[0])

if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname):
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

frame_rate_calc = 1
freq = cv2.getTickFrequency()

try:
    videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
except Exception as e:
    print(f"Failed to start video stream: {e}")
    exit(1)

time.sleep(1)

while True:
    t1 = cv2.getTickCount()

    frame1 = videostream.read()
    
    if frame1 is None:
        print("Failed to read frame")
        continue

    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    for i in range(len(scores)):
        if (scores[i] > min_conf_threshold) and (scores[i] <= 1.0) and labels[int(classes[i])] == 'cell phone':
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))
            
            # Calculate center coordinates
            center_x = int((xmin + xmax) / 2)
            center_y = int((ymin + ymax) / 2)

            # Print center coordinates
            print(f"Center coordinates for cell phone {i+1}: ({center_x}, {center_y})")
            
            current_position_x = move_servo_x(current_position_x, center_x)
            current_position_y = move_servo_y(current_position_y, center_y)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i] * 100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Draw center point (optional)
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)  # -1 to fill the circle

    
    cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Object detector', frame)

    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    if cv2.waitKey(1) == ord('q'):
        break

pwm_x.stop()
cv2.destroyAllWindows()
videostream.stop()