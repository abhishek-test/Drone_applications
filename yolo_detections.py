import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer
from cap_from_youtube import cap_from_youtube
from deep_sort_realtime.deepsort_tracker import DeepSort

youtube_url = 'https://www.youtube.com/watch?v=eEr56MfFP6I&t=315s'
cam = cap_from_youtube(youtube_url)
#cam = cv2.VideoCapture('C:\\Abhishek_Data\\My_Data\\Datasets\\videos\\video.mp4')

model = torch.hub.load('C:\\Abhishek_Data\\My_Data\\VSC_Workspace\\Python\\yolov5', \
                       'custom', path='C:\\Abhishek_Data\\Tunga_Project\\models_data\\best.pt', source='local')

model.conf = 0.25
model.iou = 0.25

fps_patch = np.zeros((30, 100, 3), dtype=np.uint8)
names = ['ped', 'ppl', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']

while(True):

    start = timer()  

    ret, frame = cam.read()    

    if ret is None:
        break

    frame = cv2.resize(frame, (1280, 720))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  

    detections = model(frame_rgb)    
    labels, cord = detections.xyxyn[0][:, -1].to('cpu').numpy(), detections.xyxyn[0][:, :-1].to('cpu').numpy()

    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    for i in range(n):
        row = cord[i]
        x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
        
        label = f"{int(row[4]*100)}"
        className = names[int(labels[i])]

        txt = className + ": " + label
        (label_width,label_height), baseline = cv2.getTextSize(txt , cv2.FONT_HERSHEY_COMPLEX, 0.3, 1)
        top_left = tuple(map(int,[int(x1),int(y1)-(label_height+baseline)]))
        top_right = tuple(map(int,[int(x1)+label_width,int(y1)]))
        org = tuple(map(int,[int(x1),int(y1)-baseline]))

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 1)
        cv2.rectangle(frame, top_left, top_right, (255,0,0), -1)
        cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_COMPLEX, 0.3, (255,255,255), 1)           

    end = timer()      
    time_ms = 1.0 / (end-start)
    
    fps_patch[:,:,:] = 0
    cv2.putText(fps_patch, "FPS : " + str(int(time_ms)), 
        (5,20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,255,255))

    frame[30:60, 1100:1200, :] = fps_patch

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)

    if(key == ord('p')):
        cv2.waitKey(0)    

    if(key == ord('q')):
        break