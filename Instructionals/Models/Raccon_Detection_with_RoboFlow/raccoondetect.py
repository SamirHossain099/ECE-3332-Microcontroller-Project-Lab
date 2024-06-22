import cv2
from roboflow import Roboflow

# Initialize the Roboflow object with an API key
rf = Roboflow(api_key="IUMgYhpwP7DC1TYcdtDX")

# Access the specific project and model version
project = rf.workspace().project("raccoon-detection-bow7l")
model = project.version(1).model

# Define the video path
video_path = "input_video.mp4" # <- Name of the video with raccoon here
output_path = "output_video.mp4" # <- Name of the processed video here

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get the video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Create a VideoWriter object
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to an image that Roboflow can process
    temp_frame_path = "temp_frame.jpg"
    cv2.imwrite(temp_frame_path, frame)
    
    # Make predictions on the frame
    try:
        prediction = model.predict(temp_frame_path, confidence=40, overlap=30)
        detections = prediction.json()['predictions']
        
        # Draw bounding boxes on the frame
        for detection in detections:
            x = int(detection['x'])
            y = int(detection['y'])
            width = int(detection['width'])
            height = int(detection['height'])
            
            # Calculate top-left corner
            start_point = (x - width // 2, y - height // 2)
            end_point = (x + width // 2, y + height // 2)
            
            # Draw the rectangle
            cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
            
            # Put the class name label
            label = detection['class']
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Write the frame to the output video
        out.write(frame)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
