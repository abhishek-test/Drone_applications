import cv2
import numpy as np
from timeit import default_timer as timer
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

    width = 640
    height = 480

    cv2.rectangle(image, (col-offset, row-offset), (col+offset, row+offset), color, 2, 8) #(x,y)
    cv2.line(image, (col, 0), (col, row-offset), color, 2, 8)
    cv2.line(image, (width-1, row), (col+offset,row), color, 2, 8)
    cv2.line(image, (col, height-1), (col, row+offset), color, 2, 8)
    cv2.line(image, (0,row), (col-offset, row), color, 2, 8)

    return image

if __name__ == '__main__':

    youtube_url = 'https://www.youtube.com/watch?v=XKlLI0KC5xo'
    #cap = cap_from_youtube(youtube_url)
    cap = cv2.VideoCapture(0)

    params = cv2.TrackerNano_Params()
    params.backbone = 'C:\\Users\\abhishek.sri\\Downloads\\nanotrack_backbone_sim.onnx'
    params.neckhead = 'C:\\Users\\abhishek.sri\\Downloads\\nanotrack_head_sim.onnx'
    tracker = cv2.TrackerNano_create(params)

    fps_patch = np.zeros((30, 100, 3), dtype=np.uint8)
    cv2.namedWindow('Frame')

    offset = 20
    width = 640
    height = 480
    x_pt = 0
    y_pt = 0
    isInitialized = False
    isClicked_b = False

    while True:

        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))

        start = timer()

        if(not isInitialized):
            
            cv2.setMouseCallback('Frame', onMouse)
            frame = drawOnImage(x_pt, y_pt, frame, offset)

            if(isClicked_b):
                box = (x_pt-offset, y_pt-offset, 2*offset, 2*offset)
                tracker.init(frame, box)
                isInitialized = True
                continue

        else:
            color = (0,255,0)
            flag, trackedBox = tracker.update(frame)
            conf = tracker.getTrackingScore()        
            frame = drawOnImage(trackedBox[0] + (int)(trackedBox[2]//2), trackedBox[1] + (int)(trackedBox[3]//2), frame, offset, color)

        end = timer()      
        fps = 1.0 / (end-start)

        fps_patch[:,:,:] = 0
        cv2.putText(fps_patch, "FPS : " + str(int(fps)), (5,20), 
            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,255,255))

        frame[30:60, 480:580, :] = fps_patch   
    
        key = cv2.waitKey(1)
        cv2.imshow("Frame", frame)
    
        if key == ord('q'):
            break

        if key == ord('r'):
            isInitialized = False
            isClicked_b = False
            offset = 20

        if key == ord('w'):
            offset = offset + 1

        if key == ord('s'):
            offset = offset - 1    

        if key == ord('p'):
            cv2.waitKey()

