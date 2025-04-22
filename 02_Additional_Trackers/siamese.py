import cv2
import numpy as np
from timeit import default_timer as timer
from cap_from_youtube import cap_from_youtube

#cap = cv2.VideoCapture('C:\\Abhishek_Data\\My_Data\\Datasets\\videos\\video.mp4')
youtube_url = 'https://www.youtube.com/watch?v=LUJ_mQzkebQ'
cap = cap_from_youtube(youtube_url)

params = cv2.TrackerNano_Params()
params.backbone = 'C:\\Users\\abhishek.sri\\Downloads\\nanotrack_backbone_sim_v2.onnx'
params.neckhead = 'C:\\Users\\abhishek.sri\\Downloads\\nanotrack_head_sim_v2.onnx'
tracker = cv2.TrackerNano_create(params)

frame_size = (1280,720)

ret, frame = cap.read()
frame = cv2.resize(frame, (1280, 720))
fps_patch = np.zeros((30, 100, 3), dtype=np.uint8)

box = cv2.selectROI("Tracker", frame)
tracker.init(frame, box)

while True:  

    ret, frame = cap.read()

    if not ret:
        break  

    frame = cv2.resize(frame, (1280, 720))

    start = timer()

    flag, trackedBox = tracker.update(frame)
    conf = tracker.getTrackingScore()

    end = timer()      
    fps = 1.0 / (end-start)
    
    fps_patch[:,:,:] = 0
    cv2.putText(fps_patch, "FPS: " + str(int(fps)), (5,20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,255,255))

    frame[30:60, 1100:1200, :] = fps_patch
    cv2.rectangle(frame, trackedBox, (0,255,255))
    cv2.putText(frame, str((int)(100*conf)), (trackedBox[0], trackedBox[1]-5),  cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,0,0))
    cv2.putText(frame, str(flag), (30, 30),  cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,0,0))

    cv2.imshow("Tracker", frame)
    
    key = cv2.waitKey(1)

    if key == ord('r'):
        box = cv2.selectROI("Tracker", frame)
        tracker.init(frame, box)

    if key == ord('q'):
        break

    if key == ord('p'):
        cv2.waitKey()