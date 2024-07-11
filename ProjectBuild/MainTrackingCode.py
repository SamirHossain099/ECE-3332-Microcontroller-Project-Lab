import cv2 as cv

def drawRectangle(frame, bbox, color=(0, 255, 0)):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv.rectangle(frame, p1, p2, color, 2)

video = cv.VideoCapture(0)  # calls facecam
#bbox = (590, 310, 100, 100) # x,y,w,h #initial bounding box location (later will be given by yolo) mac center
ret, frame = video.read()
print(frame.shape[:2])
y=(frame.shape[0]//2)-50
x=(frame.shape[1]//2)-50
bbox = ((x),(y),100,100)
if not ret:
    print("Error: Could not read initial frame")
    video.release()
    cv.destroyAllWindows()
    exit()
tracker = cv.legacy.TrackerCSRT.create()
initial = tracker.init(frame, bbox)
print("BOOSTING Tracker Initialized...")
last_known_bbox = bbox #Use last_known_bbox for aiming system
TrackingBool = True #TrackingBool indicates whether tracking is working successfully
tracking_active = True #Tracking active indicate if tracking is enabled (mostly for testing purposes)
while True:
    CamBool, frame = video.read()
    if not CamBool:
        print("Camera Feed Failed")
        break
    if tracking_active:
        TrackingBool, bbox = tracker.update(frame)
        if TrackingBool:
            last_known_bbox = bbox
            drawRectangle(frame, bbox, (0, 255, 0)) 
            print(bbox)
        else:
            print("Tracking Failed")
            tracking_active = False
    else:
        drawRectangle(frame, last_known_bbox, (0, 0, 255))  
        cv.putText(frame, "Tracking Paused", (100, 80), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv.imshow("Tracking", frame)
    if cv.waitKey(1) & 0xFF == ord('q'): #q key quits program
        break
    if cv.waitKey(1) & 0xFF == ord('r'): #r key pauses tracking - you have to hold the key down a little bit
        tracking_active = not tracking_active
cv.destroyAllWindows()
video.release()


