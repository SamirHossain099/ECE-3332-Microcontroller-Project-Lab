import cv2

video = cv2.VideoCapture(0)
status, frame1 = video.read()
status, frame2 = video.read()
delay = 0
while video.isOpened():

    #status is a bool flag which indicates if frame was read properly
    gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (15,15),0)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2,(15,15),0)
    diff = cv2.absdiff(gray2,gray)
    threshold = cv2.threshold(diff,20,255,cv2.THRESH_BINARY)[1]
    threshold = cv2.dilate(threshold, None, iterations = 2)
    contours, res = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    largest_contour = 0.0
    for contour in contours:
        if cv2.contourArea(contour) < 5500:
            continue
        if cv2.contourArea(contour) < largest_contour:
            continue
        largest_contour = cv2.contourArea(contour)
        (x,y,w,h) = cv2.boundingRect(contour)
    
        cv2.rectangle(frame1, (x,y),(x+w,y+h),(0,255,0),3)


    cv2.imshow("Video",frame1)
    delay = delay+1
    if delay == 2:
        delay = 0
        frame1 = frame2
        status, frame2 = video.read()

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()