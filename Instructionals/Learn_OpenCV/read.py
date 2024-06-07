import cv2 as cv
#Reading Images
# img = cv.imread('/Users/nicholasrethans/Documents/Photos/yus.jpg') #takes path of image and returns matrix of pixels
# cv.imshow('Friends',img) #displays image in a window 
# cv.waitKey(0) #waits infinately for a keyboard press to close window.

#Reading Videos
capture = cv.VideoCapture(0) # use num to select devices, and file path for video
while True:
    isTrue, frame = capture.read()
    #capture.read() returns a frame (to frame) and a boolean(isTrue) that says if the frame was successfully read in or not
    cv.imshow('Video',frame)
    if(cv.waitKey(20) & 0xFF==ord('d')):
        break
capture.release()
cv.destroyAllWindows()