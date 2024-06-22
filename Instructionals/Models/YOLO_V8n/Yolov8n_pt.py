## Its set to use the Pi Camera, to use your webcame use opencv's capture

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

# Initialize the Picamera2
try:
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (1280, 720)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()
except Exception as e:
    print(f"Error initializing camera: {e}")
    exit(1)

# Load the YOLOv8 model
try:
    model = YOLO("yolov8n.pt") # Choose the model to use
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    picam2.stop()
    exit(1)

try:
    while True:
        # Captures frame-by-frame
        frame = picam2.capture_array()

        # Run YOLOv8 inference on the frame
        results = model(frame)

        for result in results:
            for obj in result.boxes:
                # Extract coordinates
                x1, y1, x2, y2 = obj.xyxy[0]  # [x_min, y_min, x_max, y_max]
                confidence = obj.conf
                class_id = obj.cls

                # Check if the detected object is a toothbrush (class ID 79)
                if class_id == 79:
                    # Calculate center coordinates
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    # Print the center coordinates and other info
                    print(f"Toothbrush detected! Confidence: {confidence}, Center Coordinates: ({center_x}, {center_y})")

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the resulting frame
        cv2.imshow("Camera", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord("q"):
            break
except Exception as e:
    print(f"Error during processing: {e}")
finally:
    cv2.destroyAllWindows()
    picam2.stop()