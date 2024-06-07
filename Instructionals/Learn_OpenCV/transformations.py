import cv2 as cv
import numpy as np

img = cv.imread('/Users/nicholasrethans/Documents/Photos/yus.jpg')

#Translation
def translate(img,x,y): 
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1],img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)
"""
-x --> Left
-y --> Up
+x -->Right
+y -->Down
"""
#Rotation
def rotate(img, angle, rotPoint=None):
    (height,width) = img.shape[:2]
    if rotPoint is None:
        rotPoint = (width//2,height//2)
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width,height)
    return cv.warpAffine(img, rotMat, dimensions)

rotated = rotate(img,45)
translated = translate(rotated,100,100)

#Flipping
flip = cv.flip(img, 0)

cv.imshow('Test',flip)
#cv.imshow('Frame',img)
cv.waitKey(0)