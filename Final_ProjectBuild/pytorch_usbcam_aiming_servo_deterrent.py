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
auto_mode_pin = 15  # GPIO12
single_mode_pin = 31  # GPIO11
trigger_pin = 33  # GPIO13

# Deterrent activation durations
strobe_duration = 5.0  # seconds
auto_fire_duration = 2.0  # seconds
single_fire_shots = 5  # number of shots

# Servo settings
pan_angle = 90
tilt_angle = 90
servo_update_interval = 0.1  # seconds between servo updates
max_step_size = 4.0  # Maximum step size for servo movement

# Camera resolution
frame_width = 1080
frame_height = 720
center_x_camera = frame_width // 2
center_y_camera = frame_height // 2

# Timers
last_detection_time = time.time()
last_servo_update_time = time.time()
reset_delay = 1.0  # seconds before resetting servos to 90 degrees
cooldown_period = 1.0  # seconds after reset before tracking starts again
buffer_flush_period = 1.0  # seconds to flush the buffer after deterrent activation

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
        pan_servo.angle = pan_angle
        tilt_servo.angle = tilt_angle
        return pan_servo, tilt_servo
    except Exception as e:
        print(f"Failed to initialize servos: {e}")
        exit(1)

def initialize_gpio():
    try:
        GPIO.setwarnings(False)
        GPIO.cleanup()
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(strobe_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(auto_mode_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(single_mode_pin, GPIO.OUT, initial=GPIO.LOW)
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

def activate_orbez_gun(mode):
    if mode == 'automatic':
        print("Activating orbez gun in automatic mode...")
        GPIO.output(auto_mode_pin, GPIO.HIGH)
        GPIO.output(trigger_pin, GPIO.HIGH)
        print("Automatic mode ON, Trigger ON")
        time.sleep(auto_fire_duration)
        GPIO.output(auto_mode_pin, GPIO.LOW)
        GPIO.output(trigger_pin, GPIO.LOW)
        print("Automatic mode OFF, Trigger OFF")
    elif mode == 'single':
        print("Activating orbez gun in single fire mode...")
        GPIO.output(single_mode_pin, GPIO.HIGH)
        for shot in range(single_fire_shots):
            GPIO.output(trigger_pin, GPIO.HIGH)
            print(f"Single shot {shot + 1}")
            time.sleep(0.1)
            GPIO.output(trigger_pin, GPIO.LOW)
            time.sleep(0.1)
        GPIO.output(single_mode_pin, GPIO.LOW)
        print("Single fire mode OFF")

# def activate_speaker():
#     print("Activating speaker...")
#     import pygame
#     pygame.mixer.init()
#     pygame.mixer.music.load("alert.mp3")
#     pygame.mixer.music.play()
#     print("Playing alert sound")

def activate_random_deterrent():
    deterrents = ['strobe_light', 'orbez_gun_automatic', 'orbez_gun_single']
    selected = random.choice(deterrents)
    if selected == 'strobe_light':
        activate_strobe_light()
    elif selected == 'orbez_gun_automatic':
        activate_orbez_gun('automatic')
    elif selected == 'orbez_gun_single':
        activate_orbez_gun('single')

def update_servo_angles(center_x, center_y, pan_servo, tilt_servo):
    global pan_angle, tilt_angle, last_servo_update_time

    error_x = center_x - center_x_camera
    error_y = center_y - center_y_camera

    if time.time() - last_servo_update_time > servo_update_interval:
        delta_pan = -error_x * 0.1
        delta_tilt = error_y * 0.1

        delta_pan = max(-max_step_size, min(max_step_size, delta_pan))
        delta_tilt = max(-max_step_size, min(max_step_size, delta_tilt))

        pan_angle += delta_pan
        tilt_angle += delta_tilt

        pan_angle = max(0, min(180, pan_angle))
        tilt_angle = max(0, min(180, tilt_angle))

        pan_servo.angle = pan_angle
        tilt_servo.angle = tilt_angle

        last_servo_update_time = time.time()

def reset_servos_to_default(pan_servo, tilt_servo):
    global pan_angle, tilt_angle
    pan_angle = 90
    tilt_angle = 90
    pan_servo.angle = pan_angle
    tilt_servo.angle = tilt_angle

def flush_camera_buffer(cap, flush_duration):
    print("Flushing camera buffer...")
    flush_start_time = time.time()
    while time.time() - flush_start_time < flush_duration:
        ret, frame = cap.read()
    print("Camera buffer flushed.")

def main():
    global last_detection_time

    pca = initialize_pca9685()
    pan_servo, tilt_servo = initialize_servos(pca)
    initialize_gpio()

    model = YOLO('Final_Project_Build_JetsonOrinNano/pytorch_model/best100.pt')
    cap = cv2.VideoCapture(2)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()

    frame_count = 0
    start_time = time.time()
    in_cooldown = False

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

            results = model.track(source=frame)
            detected_object = False

            for result in results:
                for box in result.boxes:
                    detected_object = True
                    last_detection_time = current_time

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    print(f"Detected Object - Center Coordinates: x: {center_x}, y: {center_y}")

                    update_servo_angles(center_x, center_y, pan_servo, tilt_servo)

                    if abs(center_x - center_x_camera) < 50 and abs(center_y - center_y_camera) < 50:
                        print("Object detected near center, activating deterrent...")
                        activate_random_deterrent()
                        flush_camera_buffer(cap, buffer_flush_period)
                        time.sleep(5)

            if not detected_object and current_time - last_detection_time > reset_delay:
                print("No object detected, resetting servos to default position.")
                reset_servos_to_default(pan_servo, tilt_servo)
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
