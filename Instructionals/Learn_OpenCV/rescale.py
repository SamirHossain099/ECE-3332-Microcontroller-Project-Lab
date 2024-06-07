import cv2 as cv
img = cv.imread('/Users/nicholasrethans/Documents/Photos/yus.jpg') #takes path of image and returns matrix of pixels
cv.imshow('Friends',img) #displays image in a window 

def rescaleFrame(frame, scale=0.75):
    #images, video, live video
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width,height)

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

def changeRes(width,height):
    #only for live video
    capture.set(3,width)
    capture.set(4,height)

capture = cv.VideoCapture(0) # use num to select devices, and file path for video
resized_img = rescaleFrame(img)
cv.imshow('Resized Friends',resized_img)

while True:
    isTrue, frame = capture.read()
    #capture.read() returns a frame (to frame) and a boolean(isTrue) that says if the frame was successfully read in or not
    
    frame_resized = rescaleFrame(frame)
    
    cv.imshow('Video',frame)
    cv.imshow('Video Resized', frame_resized)
    if(cv.waitKey(20) & 0xFF==ord('d')):
        break
capture.release()
cv.destroyAllWindows()
cv.waitKey(0) #waits infinately for a keyboard press to close window.