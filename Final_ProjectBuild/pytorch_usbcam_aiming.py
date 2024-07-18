import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('Final_Project_Build_JetsonOrinNano/pytorch_model/best100.pt')

# OpenCV VideoCapture for the webcam
cap = cv2.VideoCapture(1)

# Set the resolution to 1080p
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break

    # Process the frame with the YOLO model
    results = model.track(source=frame)
    
    # Loop through detections and draw bounding boxes
    for result in results:
        for box in result.boxes:
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

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
