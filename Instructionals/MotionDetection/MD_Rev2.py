"""
Motion detection with cropping and resizing

New: Aim turning angle
"""
import cv2 as cv
import numpy as np
import math

def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    # If the bounding boxes are integers, convert them to float.
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(area)  # Sort by area instead of confidence score

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")
def center_distance(frame, bbox):
    width, height = frame.shape[:2]
    centerFrameY = width//2
    centerFrameX = height//2
    x,y,w,h = bbox
    centerBBoxX = (x+(w//2))
    centerBBoxY = (y+(h//2))

    xDist = centerBBoxX - centerFrameX
    yDist = centerBBoxY - centerFrameY
    return abs(xDist), abs(yDist)
def calculate_angles(frame_center, bbox_center, distance):
    frame_center_x, frame_center_y = frame_center
    bbox_center_x, bbox_center_y = bbox_center
    x_dist, y_dist = distance

    x_angle_rad = math.atan2(x_dist, frame_center_y - bbox_center_y)
    y_angle_rad = math.atan2(y_dist, frame_center_x - bbox_center_x)
    x_angle_deg = math.degrees(x_angle_rad)
    y_angle_deg = math.degrees(y_angle_rad)
    return x_angle_deg, y_angle_deg

cap = cv.VideoCapture(0)
ret, frame1 = cap.read()
ret, frame2 = cap.read()
width, height = frame1.shape[:2]
centerX = height//2 #be careful, opencv just calls the smaller dimension the width which doesnt necessarily mean x axis
centerY = width//2
print(width,height)
print(centerX, centerY)
while cap.isOpened():
    diff = cv.absdiff(frame1,frame2) #finds difference between two frames
    gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY) #easier to find contours in grayscale mode
    blur = cv.GaussianBlur(gray, (5,5), 0)
    #_,thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
    thresh = cv.adaptiveThreshold(blur,40,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY, 47,1)
    dilated = cv.dilate(thresh, None, iterations=3)
    contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #cv.drawContours(frame1, contours, -1, (255,0,0), 2) #just display contours on screen
    for contour in contours:
        #save coordinates of contours
        (x,y,w,h) = cv.boundingRect(contour)
        #find area of contour and say if area is smaller than area of a person, we dont want to draw a rectangle
        #if area larger than a persons area, draw a rectangle.
        if cv.contourArea(contour) < 700:
            continue
        cv.rectangle(frame1, (x,y),(x+w,y+h),(255,0,0),thickness=3)

        #Cropping
        crop = frame1[y:y+h, x:x+w]
        if (crop.shape[0:2] != (0,0)) & (w!=1280)&(h!=720): #check to make sure crop is not zero to prevent resized zero error and the entire screen isnt interpreted as a contour
            resized = cv.resize(crop, (400,400), interpolation=cv.INTER_AREA)
            #cv.imshow("cropped and resized",resized)
        cv.putText(frame1,"Status: {}".format('Movement'),(10,20),cv.FONT_HERSHEY_SIMPLEX,1, (0,0,255),3)
        
        #Center Distance
        #xDiff,yDiff = center_distance(frame1,cv.boundingRect(contour))
        #print(xDiff,yDiff)
        cv.line(frame1, (centerX,centerY), (x+w//2,y+h//2),(0,255,0),thickness=1)

    cv.imshow("feed",frame1)
    frame1=frame2
    ret, frame2 = cap.read()
    if(cv.waitKey(20) & 0xFF==ord('d')):
        break
cv.destroyAllWindows()
cap.release()