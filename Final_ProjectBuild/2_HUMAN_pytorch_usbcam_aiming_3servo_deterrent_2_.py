import time
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
import cv2
from ultralytics import YOLO
import Jetson.GPIO as GPIO
import random

# GPIO Pins
strobe_pin = 7  # GPIO9
trigger_pin = 33  # GPIO13

# Deterrent activation durations
strobe_duration = 2.0  # seconds
shot_duration = 1.0
shot_delay = 0.25  # seconds between shots
num_shots = 1  # number of shots

# Servo settings
pan_angle = 90
tilt_angle = 90
tilt_angle_opposite = 90
servo_update_interval = 0.1  # seconds between servo updates
max_step_size = 1.0  # Maximum step size for servo movement

# Camera resolution
frame_width = 1080
frame_height = 720
center_x_camera = frame_width // 2
center_y_camera = frame_height // 2

# Timers
last_detection_time = time.time()
last_servo_update_time = time.time()
reset_delay = 3.0  # seconds before resetting servos to 90 degrees
cooldown_period = 5.0  # seconds after shooting before tracking next human
buffer_flush_period = 1.0  # seconds to flush the buffer after deterrent activation

# Tilt angle limits
tilt_min_angle = 25  # Minimum tilt angle to prevent tilting down
tilt_max_angle = 145  # Maximum tilt angle to prevent tilting up too far

# Track only humans
human_class_id = 0  # Assuming 'person' class ID is 0

def initialize_pca9685():
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c)
        pca.frequency = 50
        return pca
    except Exception as e:
        print(f"Failed to initialize PCA9685: {e}")
        exit(1)

def initialize_servos(pca):
    try:
        pan_servo = servo.Servo(pca.channels[0], min_pulse=500, max_pulse=2500)
        tilt_servo = servo.Servo(pca.channels[1], min_pulse=500, max_pulse=2500)
        tilt_servo_opposite = servo.Servo(pca.channels[2], min_pulse=500, max_pulse=2500)
        pan_servo.angle = pan_angle
        tilt_servo.angle = tilt_angle
        tilt_servo_opposite.angle = tilt_angle_opposite
        return pan_servo, tilt_servo, tilt_servo_opposite
    except Exception as e:
        print(f"Failed to initialize servos: {e}")
        exit(1)

def initialize_gpio():
    try:
        GPIO.setwarnings(False)
        GPIO.cleanup()
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(strobe_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(trigger_pin, GPIO.OUT, initial=GPIO.LOW)
    except Exception as e:
        print(f"Failed to initialize GPIO: {e}")
        exit(1)

def activate_strobe_light():
    print("Activating strobe light...")
    GPIO.output(strobe_pin, GPIO.HIGH)
    print("Strobe light ON")
    time.sleep(strobe_duration)
    GPIO.output(strobe_pin, GPIO.LOW)
    print("Strobe light OFF")

def activate_orbez_gun():
    print("Activating orbez gun...")
    for shot in range(num_shots):
        GPIO.output(trigger_pin, GPIO.HIGH)
        print(f"Firing shot {shot + 1}")
        time.sleep(shot_duration)
        GPIO.output(trigger_pin, GPIO.LOW)
        time.sleep(shot_delay)
    print("Orbez gun activation complete")

def activate_random_deterrent():
    deterrents = ['orbez_gun']
    selected = random.choice(deterrents)
    if selected == 'strobe_light':
        activate_strobe_light()
    elif selected == 'orbez_gun':
        activate_orbez_gun()

def update_servo_angles(center_x, center_y, pan_servo, tilt_servo, tilt_servo_opposite):
    global pan_angle, tilt_angle, tilt_angle_opposite, last_servo_update_time

    error_x = center_x - center_x_camera
    error_y = center_y - center_y_camera

    if time.time() - last_servo_update_time > servo_update_interval:
        delta_pan = -error_x * 0.1
        delta_tilt = error_y * 0.1

        delta_pan = max(-max_step_size, min(max_step_size, delta_pan))
        delta_tilt = max(-max_step_size, min(max_step_size, delta_tilt))

        pan_angle += delta_pan
        tilt_angle += delta_tilt
        tilt_angle_opposite -= delta_tilt

        # Clamp angles to valid ranges
        if pan_angle < 0 or pan_angle > 180 or tilt_angle < tilt_min_angle or tilt_angle > tilt_max_angle or tilt_angle_opposite < tilt_min_angle or tilt_angle_opposite > tilt_max_angle:
            print("Resetting servos to default due to out-of-range angle.")
            reset_servos_to_default(pan_servo, tilt_servo, tilt_servo_opposite)
        else:
            pan_servo.angle = pan_angle
            tilt_servo.angle = tilt_angle
            tilt_servo_opposite.angle = tilt_angle_opposite

        last_servo_update_time = time.time()

def reset_servos_to_default(pan_servo, tilt_servo, tilt_servo_opposite):
    global pan_angle, tilt_angle, tilt_angle_opposite
    pan_angle = 90
    tilt_angle = 90
    tilt_angle_opposite = 90
    pan_servo.angle = pan_angle
    tilt_servo.angle = tilt_angle
    tilt_servo_opposite.angle = tilt_angle_opposite
    print(f"Servos reset to default: pan_angle={pan_angle}, tilt_angle={tilt_angle}, tilt_angle_opposite={tilt_angle_opposite}")

def main():
    global last_detection_time

    pca = initialize_pca9685()
    pan_servo, tilt_servo, tilt_servo_opposite = initialize_servos(pca)
    initialize_gpio()

    model = YOLO('yolov8n.pt')  # Model path
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()

    frame_count = 0
    start_time = time.time()
    in_cooldown = False
    last_target_center = None

    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame.")
                break

            current_time = time.time()
            if in_cooldown:
                if current_time - last_detection_time < cooldown_period:
                    print("In cooldown period, skipping tracking.")
                    cv2.imshow('Object Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                else:
                    in_cooldown = False
                    last_target_center = None

            results = model.track(source=frame, conf=0.60)
            detected_object = False
            closest_distance = float('inf')
            closest_center = None

            for result in results:
                for box in result.boxes:
                    if box.cls == human_class_id:
                        detected_object = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        distance = ((center_x - center_x_camera) ** 2 + (center_y - center_y_camera) ** 2) ** 0.5
                        if distance < closest_distance and (last_target_center is None or distance != ((last_target_center[0] - center_x_camera) ** 2 + (last_target_center[1] - center_y_camera) ** 2) ** 0.5):
                            closest_distance = distance
                            closest_center = (center_x, center_y)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                        print(f"Detected Human - Center Coordinates: x: {center_x}, y: {center_y}")

            if detected_object and closest_center is not None:
                last_detection_time = current_time
                update_servo_angles(closest_center[0], closest_center[1], pan_servo, tilt_servo, tilt_servo_opposite)

                if abs(closest_center[0] - center_x_camera) < 50 and abs(closest_center[1] - center_y_camera) < 50:
                    print("Human detected near center, activating deterrent...")
                    activate_random_deterrent()
                    last_target_center = closest_center
                    in_cooldown = True

            if not detected_object and current_time - last_detection_time > reset_delay:
                print("No human detected, resetting servos to default position.")
                reset_servos_to_default(pan_servo, tilt_servo, tilt_servo_opposite)
                last_detection_time = current_time  # Reset timer to prevent immediate reset again
                in_cooldown = True

            cv2.imshow('Object Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            print("FPS: {:.2f}".format(fps))

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()

if __name__ == '__main__':
    main()
