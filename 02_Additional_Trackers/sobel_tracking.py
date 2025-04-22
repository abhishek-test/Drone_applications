import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer
from cap_from_youtube import cap_from_youtube

youtube_url = 'https://www.youtube.com/watch?v=XKlLI0KC5xo'
cam = cap_from_youtube(youtube_url)

str_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5), (-1,-1))
drawing = np.zeros((720, 1280, 3), dtype=np.uint8)

while(True):

    start = timer()

    _, frame = cam.read()
    frame = cv2.resize(frame, (640, 360))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    #grad = grad*5
    
    _, thresh = cv2.threshold(grad, 50, 255, cv2.THRESH_BINARY, None)
    #filtered = cv2.erode(grad, str_kernel, None, (-1, -1))
    filtered = cv2.dilate(thresh, str_kernel, None, (-1, -1))    

    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        
    for i in range(len(contours)):
        width = boundRect[i][2]
        height = boundRect[i][3]    
        top_x = boundRect[i][1] 
        area = width * height

        if ( (width < 320) and (height < 120) and (top_x < 240) ):
            cv2.rectangle(frame, (int(boundRect[i][0]), int(boundRect[i][1])), \
                (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), (0,0,255), 2)


    end = timer()      
    time_ms = 1.0 / (end-start)

    cv2.putText(frame, "FPS : " + str(int(time_ms)), (30, 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,255))

    drawing[0:360, 0:640, :] =  cv2.cvtColor(abs_grad_x, cv2.COLOR_GRAY2BGR)
    drawing[0:360, 640:1280, :] =  cv2.cvtColor(abs_grad_y, cv2.COLOR_GRAY2BGR)
    drawing[360:720, 0:640, :] =  cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    drawing[360:720, 640:1280, :] =  frame

    cv2.imshow("Object Detection", frame)
    #cv2.imshow("Motion Mask", filtered)

    key = cv2.waitKey(1)

    if(key == ord('p')):
        cv2.waitKey(0)

    if(key == ord('q')):
        break