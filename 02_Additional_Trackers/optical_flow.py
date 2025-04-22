import numpy as np
import cv2 
import argparse
from cap_from_youtube import cap_from_youtube

def onMouse(event, x, y, flags, param):
    global x_pt, y_pt, isClicked_b
    if (event == cv2.EVENT_MOUSEMOVE):
        x_pt = x
        y_pt = y
    
    if (event == cv2.EVENT_LBUTTONDBLCLK):
        x_pt = x
        y_pt = y
        isClicked_b = True


def drawOnImage(col, row, image, offset, color=(0,255,255)):

    width = 1280 #640
    height = 720 #480

    cv2.rectangle(image, (col-offset, row-offset), (col+offset, row+offset), color, 2, 8) #(x,y)
    cv2.line(image, (col, 0), (col, row-offset), color, 2, 8)
    cv2.line(image, (width-1, row), (col+offset,row), color, 2, 8)
    cv2.line(image, (col, height-1), (col, row+offset), color, 2, 8)
    cv2.line(image, (0,row), (col-offset, row), color, 2, 8)

    return image



youtube_url = 'https://www.youtube.com/watch?v=LUJ_mQzkebQ'
#cap = cap_from_youtube(youtube_url)
cap = cv2.VideoCapture("C:\\Abhishek_Data\\My_Data\\Datasets\\videos\\video.mp4")

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7)

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15), maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

ret, frame = cap.read()
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
old_gray = frame_gray

p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
p1 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

good_new = p1
good_old = p0

width = 1280 #640
height = 720 #480

mask = np.zeros((height, width, 3), dtype=np.uint8)
cv2.namedWindow('Frame')

offset = 20
x_pt = 0
y_pt = 0
isInitialized = False
isClicked_b = False

while(1):

    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        print('No frames grabbed!')
        break

    if(not isInitialized):
            
        cv2.setMouseCallback('Frame', onMouse)
        frame = drawOnImage(x_pt, y_pt, frame, offset)

        if(isClicked_b):
            
            box = (x_pt-offset, y_pt-offset, 2*offset, 2*offset)     
            old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #mask_bbox = np.zeros((480, 640, 3), dtype=np.uint8)
            mask_bbox = np.zeros_like(old_gray)
            mask_bbox[ y_pt-offset:y_pt+offset, x_pt-offset:x_pt+offset] = 1

            p0 = cv2.goodFeaturesToTrack(old_gray, mask=mask_bbox, **feature_params)

            '''
            old_gray_black = np.zeros_like(old_gray)
            old_gray_black[y_pt-offset:y_pt+offset, x_pt-offset:x_pt+offset] = old_gray[y_pt-offset:y_pt+offset, x_pt-offset:x_pt+offset]            
            p0 = cv2.goodFeaturesToTrack(old_gray_black, mask = None, **feature_params)
            '''
            isInitialized = True
            continue

    else:

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            #mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            

    img = cv2.add(frame, mask)
    cv2.imshow('Frame', img)
    
    k = cv2.waitKey(20)
    if k == ord('q'):
        break

    if k == ord('r'):
        isInitialized = False
        mask = np.zeros((height, width, 3), dtype=np.uint8)
        x_pt = 0
        y_pt = 0
        offset = 20
        isClicked_b = False


    if k == ord('w'):
        offset = offset + 1

    if k == ord('s'):
        offset = offset - 1

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()