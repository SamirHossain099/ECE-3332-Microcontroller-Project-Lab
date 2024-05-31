import numpy as np
import cv2

cap = cv2.VideoCapture(0) #zero is the webcam number
                          #zero can be replaced by a video path too
while True: 
    ret, frame = cap.read() #creating a window frame to read the capture to 
    width = int(cap.get(3)) #acquiring dimension characteristics by indexing the object's fields
    height = int(cap.get(4))

    image = np.zeros(frame.shape, np.uint8) #creating a empty array the size of the frame

    smaller_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) #shrink frame to fit 4 into one frame
    image[:height//2, :width//2] = smaller_frame #top left
    image[height//2:, :width//2] = smaller_frame #bottom left
    image[:height//2, width//2:] = smaller_frame #top right
    image[height//2:, width//2:] = smaller_frame #bottom right

    image[:height//2, :width//2] = cv2.rotate(smaller_frame,cv2.ROTATE_180) #top left
    image[height//2:, :width//2] = cv2.rotate(smaller_frame,cv2.ROTATE_180) #bottom left
    image[:height//2, width//2:] = smaller_frame #top right
    image[height//2:, width//2:] = smaller_frame #bottom right

    cv2.imshow('frame',image)       #display the frame 
    if cv2.waitKey(1) == ord('q'):  #sets the key "q" as the key to close the frame
        break

cap.release()           #releases webcam capture device so other programs can use
cv2.destroyAllWindows() #cleans memory