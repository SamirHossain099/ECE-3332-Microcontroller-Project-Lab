import cv2 as cv
import numpy as np
import math
blank = np.zeros((500,500,3), dtype='uint8') #uint8 is the data type used for images
                #(height,width,rgb color channels) 
#paint image a certain color
#blank[200:300,200:300] = 0,0,255

#draw rectangle
cv.rectangle(blank, (0,0), (blank.shape[1]//2,blank.shape[0]//2), (0,255,0),thickness=cv.FILLED)

#draw circle
cv.circle(blank, (blank.shape[1]//2,blank.shape[0]//2), 40, (255,0,0), thickness=3)

#draw line
cv.line(blank, (100,100),(200,400),(12,134,52),thickness=3)

#text on an image
cv.putText(blank, 'Hello',(255,255),cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,0,255), 2)

cv.imshow('Frame',blank)
cv.waitKey(0)