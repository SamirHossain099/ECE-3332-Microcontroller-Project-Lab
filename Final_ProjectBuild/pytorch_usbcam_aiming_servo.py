import time
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
import cv2
from ultralytics import YOLO

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

# Load the YOLO model
model = YOLO('Final_Project_Build_JetsonOrinNano/pytorch_model/yolov8n.pt')

# OpenCV VideoCapture for the webcam
cap = cv2.VideoCapture(2)

# Set the resolution to 720p
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Camera resolution
frame_width = 1080
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
max_step_size = 4.0  # Maximum step size for servo movement

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break

    # Process the frame with the YOLO model
    results = model.track(source=frame)
    detected_object = False

    # Loop through detections and draw bounding boxes
    for result in results:
        for box in result.boxes:
            detected_object = True
            last_detection_time = time.time()

            # Get the coordinates of the bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw the center point
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Print the center coordinates of the bounding box
            print(f"Detected Object - Center Coordinates: x: {center_x}, y: {center_y}")

            # Calculate the difference between the center of the bounding box and the center of the camera
            error_x = center_x - center_x_camera
            error_y = center_y - center_y_camera

            # Update servos if enough time has passed since the last update
            if time.time() - last_servo_update_time > servo_update_interval:
                # Calculate the desired angle changes
                delta_pan = -error_x * 0.1  # Adjust the multiplier for sensitivity
                delta_tilt = error_y * 0.1  # Adjust the multiplier for sensitivity

                # Limit the angle changes to the maximum step size
                delta_pan = max(-max_step_size, min(max_step_size, delta_pan))
                delta_tilt = max(-max_step_size, min(max_step_size, delta_tilt))

                # Adjust servo angles
                pan_angle += delta_pan
                tilt_angle += delta_tilt

                # Clamp the angles to valid ranges
                if pan_angle < 0 or pan_angle > 180 or tilt_angle < 0 or tilt_angle > 180:
                    pan_angle = 90
                    tilt_angle = 90

                # Move the servos
                pan_servo.angle = pan_angle
                tilt_servo.angle = tilt_angle

                last_servo_update_time = time.time()

    if not detected_object and time.time() - last_detection_time > reset_delay:
        # Reset servos to center position if no object is detected for a certain time
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
