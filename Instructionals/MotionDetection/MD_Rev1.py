import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
ret, frame1 = cap.read()
ret, frame2 = cap.read()
#ret, frame3 = cap.read()

while cap.isOpened():
    diff = cv.absdiff(frame1,frame2) #finds difference between two frames
    gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY) #easier to find contours in grayscale mode
    blur = cv.GaussianBlur(gray, (21,21), 0)
    _,thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
    #thresh = cv.adaptiveThreshold(blur,20,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY, 47,1)
    dilated = cv.dilate(thresh, None, iterations=3)
    contours, _ = cv.findContours(dilated, cv.CHAIN_APPROX_SIMPLE, cv.CHAIN_APPROX_SIMPLE) #change RETR_TREE to CHAIN_APPROX_SIMPLE
    #cv.drawContours(frame1, contours, -1, (255,0,0), 2) #just display contours on screen
    cv.drawContours(frame1,contours, -1, (0,255,0),2)
    for contour in contours:
        #save coordinates of contours
        (x,y,w,h) = cv.boundingRect(contour)
        #find area of contour and say if area is smaller than area of a person, we dont want to draw a rectangle
        #if area larger than a persons area, draw a rectangle.
        if cv.contourArea(contour) < 2500:
            continue
        cv.rectangle(frame1, (x,y),(x+w,y+h),(255,0,0),thickness=2)
        #crop = frame1[y:y+h, x:x+w]
        #if crop.shape[0:2] != (0,0): #check to make sure crop is not zero to prevent resized zero error
        #    resized = cv.resize(crop, (400,400), interpolation=cv.INTER_AREA)
            #cv.imshow("cropped and resized",resized)
        #cv.putText(frame1,"Status: {}".format('Movement'),(10,20),cv.FONT_HERSHEY_SIMPLEX,1, (0,0,255),3)
    #Contour overlay, but no noise removal at this point
    #cv.imshow("Contours",contours)
    cv.imshow("BBox Overlay",frame1)
    frame1=frame2
    ret, frame2 = cap.read()
    if(cv.waitKey(20) & 0xFF==ord('d')):
        break
cv.destroyAllWindows()
cap.release()