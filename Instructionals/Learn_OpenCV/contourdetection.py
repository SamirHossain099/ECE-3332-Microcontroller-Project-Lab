import cv2 as cv
import numpy as np

img = cv.imread('/Users/nicholasrethans/Documents/Photos/yus.jpg')

blank = np.zeros(img.shape, 'uint8')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)

canny = cv.Canny(blur, 125,175)

ret, thresh = cv.threshold(gray,125,255,cv.THRESH_BINARY)

#contours
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} found')

cv.drawContours(blank, contours, -1, (0,0,255),2)


cv.imshow('Test',thresh)
cv.imshow('Frame',blank)
cv.waitKey(0)