import cv2
import numpy as np
import time
from jetson_inference import detectNet
from jetson_utils import cudaFromNumpy
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

# Initialize the object detection network
net = detectNet("ssd-mobilenet-v2", threshold=0.5)

# Class ID for cellphone (77 COCO dataset)
cellphone_class_id = 77

# GStreamer pipeline for USB webcam using MJPEG format
gst_pipeline = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg, width=1280, height=720, framerate=30/1 ! "
    "jpegdec ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
)

# OpenCV VideoCapture with GStreamer pipeline
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Initialize I2C bus and PCA9685
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

# Create servo objects for pan (channel 0) and tilt (channel 1)
pan_servo = servo.Servo(pca.channels[0], min_pulse=500, max_pulse=2500)
tilt_servo = servo.Servo(pca.channels[1], min_pulse=500, max_pulse=2500)

# Initialize servo angles
pan_angle = 90
tilt_angle = 90
pan_servo.angle = pan_angle
tilt_servo.angle = tilt_angle

# Camera resolution
frame_width = 1280
frame_height = 720
center_x_camera = frame_width // 2
center_y_camera = frame_height // 2

# Variables to measure FPS
frame_count = 0
start_time = time.time()

# Variables for servo control
last_detection_time = time.time()
last_servo_update_time = time.time()
reset_delay = 2.0  # seconds before resetting servos to 90 degrees
servo_update_interval = 0.1  # seconds between servo updates

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to RGBA
    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    
    # Convert frame to CUDA image
    img_cuda = cudaFromNumpy(frame_rgba)
    
    detections = net.Detect(img_cuda)
    detected_cellphone = False

    # Loop through detections and draw bounding boxes for cellphones
    for detection in detections:
        if detection.ClassID == cellphone_class_id:
            detected_cellphone = True
            last_detection_time = time.time()

            # Get the coordinates of the bounding box
            left = int(detection.Left)
            top = int(detection.Top)
            right = int(detection.Right)
            bottom = int(detection.Bottom)

            # Calculate the center coordinates
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2

            # Draw the bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw the label
            label = "Cellphone"
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Print the center coordinates of the bounding box
            print("Detected Cellphone - Center Coordinates: x: {}, y: {}".format(center_x, center_y))

            # Calculate the difference between the center of the bounding box and the center of the camera
            error_x = center_x - center_x_camera
            error_y = center_y - center_y_camera

            # Update servos if enough time has passed since the last update
            if time.time() - last_servo_update_time > servo_update_interval:
                # Adjust servo angles based on the error
                pan_angle -= error_x * 0.1  # Multiplier for sensitivity
                tilt_angle += error_y * 0.1  # Multiplier for sensitivity

                # Clamp the angles to valid ranges
                if pan_angle < 0 or pan_angle > 180 or tilt_angle < 0 or tilt_angle > 180:
                    pan_angle = 90
                    tilt_angle = 90

                # Move the servos
                pan_servo.angle = pan_angle
                tilt_servo.angle = tilt_angle

                last_servo_update_time = time.time()

    if not detected_cellphone and time.time() - last_detection_time > reset_delay:
        # Reset servos to center position if no cellphone is detected for a certain time
        pan_angle = 90
        tilt_angle = 90
        pan_servo.angle = pan_angle
        tilt_servo.angle = tilt_angle

    # Display the frame with OpenCV
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update the frame count
    frame_count += 1
    
    # Calculate the elapsed time
    elapsed_time = time.time() - start_time
    
    # Calculate FPS
    if elapsed_time > 0:
        fps = frame_count / elapsed_time
    else:
        fps = 0
    
    print("FPS: {:.2f}".format(fps))

cap.release()
cv2.destroyAllWindows()
