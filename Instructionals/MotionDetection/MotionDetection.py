import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    diff = cv.absdiff(frame1,frame2) #finds difference between two frames
    gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY) #easier to find contours in grayscale mode
    blur = cv.GaussianBlur(gray, (5,5), 0)
    _,thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
    dilated = cv.dilate(thresh, None, iterations=3)
    contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #cv.drawContours(frame1, contours, -1, (255,0,0), 2) #just display contours on screen
    for contour in contours:
        #save coordinates of contours
        (x,y,w,h) = cv.boundingRect(contour)
        #find area of contour and say if area is smaller than area of a person, we dont want to draw a rectangle
        #if area larger than a persons area, draw a rectangle.
        if cv.contourArea(contour) < 800:
            continue
        cv.rectangle(frame1, (x,y),(x+w,y+h),(255,0,0),thickness=2)
        cv.putText(frame1,"Status: {}".format('Movement'),(10,20),cv.FONT_HERSHEY_SIMPLEX,1, (0,0,255),3)

    #Contour overlay, but no noise removal at this point


    cv.imshow("feed",frame1)
    frame1=frame2
    ret, frame2 = cap.read()
    if(cv.waitKey(20) & 0xFF==ord('d')):
        break
cv.destroyAllWindows()
cap.release()