import cv2 as cv

img = cv.imread('/Users/nicholasrethans/Documents/Photos/yus.jpg')

#Grayscale img
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Blur
blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
                            #must be odd numbers (controls amt of blur)

#Edge Cascade (Find the edges present in an image)
canny = cv.Canny(blur, 125, 175)
"""
You can play with feeding the Canny function a crisp or more blurred image for it to detect more or less edges
"""
#Dillating the image
dilated = cv.dilate(canny, (5,5), iterations=5)

#Eroding
eroded = cv.erode(dilated, (5,5), iterations = 5)

#resize
resized = cv.resize(img, (500,500))

#Cropping
cropped = img[50:200, 200:400]

cv.imshow('Test',cropped)
cv.imshow('Frame',img)
cv.waitKey(0)