# Import Modules
import cv2 as cv
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import urllib

video_file = "/home/nrethans/Downloads/Test Car Video.mp4"
def drawRectangle(frame, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
            # x             y
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            # x+w                       y+h
    cv.rectangle(frame, p1, p2, (0,255,0), 1)

def displayRectangle(frame, bbox):
    plt.figure(figsize=(20,10))
    frameCopy = frame.copy()
    drawRectangle(frameCopy, bbox)
    frameCopy = cv.cvtColor(frameCopy, cv.COLOR_RGB2BGR)
    plt.imshow(frameCopy); plt.axis('off')

def drawText(frame, txt, location, color = (50,170,50)):
    cv.putText(frame, txt, location, cv.FONT_HERSHEY_SIMPLEX, 1, color, 3)

# Creating Tracker Instance
tracker_types = ['BOOSTING','MIL','KCF','CSRT','TLD','MEDIANFLOW','GOTURN','MOSSE']

#Change the index to change the tracker type
tracker_type = tracker_types[7]

if tracker_type == 'BOOSTING':
    tracker = cv.legacy.TrackerBoosting.create()
elif tracker_type == 'MIL':
    tracker = cv.TrackerMIL.create()
elif tracker_type == 'KCF':
    tracker = cv.TrackerKCF.create()
elif tracker_type == 'CSRT':
    tracker = cv.legacy.TrackerCSRT.create()
elif tracker_type == 'TLD':
    tracker = cv.legacy.TrackerTLD.create()
elif tracker_type == 'MEDIANFLOW':
    tracker= cv.legacy.TrackerMedianFlow.create()
elif tracker_type == 'GOTURN': #requires weights I dont have downloaded
    tracker = cv.TrackerGOTURN.create()
elif tracker_type == 'MOSSE':
    tracker = cv.legacy.TrackerMOSSE.create()
else:
    tracker = cv.legacy.TrackerMOSSE.create()

#read video (can change to webcam)
video = cv.VideoCapture(video_file)
_,frame = video.read()
if not video.isOpened():
    print("Could not open video")
    sys.exit()
else:
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(width)
    print(height)

video_file = 'race_car'+tracker_type+'.mp4'
video_out = cv.VideoWriter(video_file,cv.VideoWriter.fourcc(*'mp4v'),10,(width,height))

#Define Bounding Box (Later Done by yolo)
bbox = (245, 150, 60, 35) #initial bounding box upper left and lower right coordinates
       #x, y, w  , h
displayRectangle(frame,bbox)

#Initialize Tracker
_ = tracker.init(frame,bbox)

#Read frames and track object
totalframes=1
totalfps=0
while True:
    _, frame = video.read()
    if not _:
        break

    # Update timer
    timer = cv.getTickCount()
    # Update tracker
    _, bbox = tracker.update(frame)
    # Calculate FPS
    fps=cv.getTickFrequency()/(cv.getTickCount()-timer)
    totalfps=totalfps+fps
    totalframes=totalframes+1
    # Draw bounding box
    if _:
        drawRectangle(frame,bbox)
    else:
        drawText(frame,"Tracking failure detected", (80,140),(0,0,225))
    # Display Info
    drawText(frame,tracker_type+"Tracker", (80,60))
    drawText(frame, "FPS : "+str(int(fps)), (80,100))
    #Write frame to video
    video_out.write(frame)
video.release()
video_out.release()
print("Average FPS: "+str(int((totalfps/totalframes))))
"""
2.5 GHz Quad Core Intel Core i7
640x360 360p resolution
1. Boosting - avg 61 fps
2. MIL - avg 19 fps
3. KCF - avg 583 fps
4. CRST - avg 34 fps
5. TLD - avg 38 fps
6. MEDIANFLOW - avg 745 fps
7. GOTURN
The only Deep Learning Based, Most Accurate - Requires Weights download (see 5:23)
8. MOSSE - avg 8997** fps

1.5 Ghz Broadcom BCM2711 SoC

1. Boosting - avg 22 fps
2. MIL - avg 7 fps
3. KCF - avg 168 fps
4. CRST - avg 12 fps
5. TLD - avg 13 fps
6. MEDIANFLOW - avg 237 fps
7. GOTURN
The only Deep Learning Based, Most Accurate - Requires Weights download (see 5:23)
8. MOSSE - avg 2826 fps

"""
