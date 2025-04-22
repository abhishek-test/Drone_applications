
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracking_legacy.hpp>

#include <string>
#include <sstream>

#include "ObjDetection.h"
#include "utility.h"
#include "trackerMosse.h"

using namespace cv; 
using namespace std;

Rect boxFromPtAndRect(Point pt, Rect box) {
    
    int xx = pt.x - (box.width/2);
    int yy = pt.y - (box.height/2);
    int width = box.width;
    int height = box.height;

    Rect newRect = Rect(xx, yy, width, height);

    return newRect;
}


void drawRect(Mat &input, Rect box, Scalar color) {
    int offset = 15;

    Point dl = Point(box.tl().x, box.tl().y + box.height);
    Point tr = Point(box.tl().x + box.width, box.tl().y);

    // top left
    line(input, box.tl(), Point(box.tl().x + offset, box.tl().y), color, 2, 8);
    line(input, box.tl(), Point(box.tl().x, box.tl().y+offset), color, 2, 8);

    // down left
    line(input, dl, Point(dl.x + offset, dl.y), color, 2, 8);
    line(input, dl, Point(dl.x, dl.y-offset), color, 2, 8);

    // bottom right
    line(input, box.br(), Point(box.br().x - offset, box.br().y), color, 2, 8);
    line(input, box.br(), Point(box.br().x, box.br().y - offset), color, 2, 8);

    // top right
    line(input, tr, Point(tr.x - offset, tr.y), color, 2, 8);
    line(input, tr, Point(tr.x, tr.y + offset), color, 2, 8);    
}


int main()
{
    // Load video
    VideoCapture capture("C:\\Abhishek_Data\\My_Data\\Datasets\\videos\\video.mp4");
    
    if (!capture.isOpened()) {
        std::cerr << "Failed to open video file!" << std::endl;
        return -1;
    }

    //VideoWriter vidOut("output.avi", VideoWriter::fourcc('M','J','P','G'), 10, Size(1280, 720));
    
    Mat frame;

    bool isTrackerInitialized_b = false;
    bool useDetections = false;
    bool isDetected = false;
    bool isObjectSelected = false;

    Rect trackedBoundingBox, trackedBoundingBoxCopy;
    Rect detectedBoundingBox;
    Rect predictedBox;

    int trackingConf = 0;
    int trackedFrameCount = 0;
    
    // Initialize Kalman filter
    KalmanFilter kalmanFilter(4, 2, 0);
    Mat measurement = Mat::zeros(2, 1, CV_32F);
    Mat prediction;
    
    // Initialize MOSSE tracker
    Ptr<Tracker> tracker = legacy::upgradeTrackingAPI(legacy::TrackerMOSSE::create());
    //MosseTracker mTracker;
    double psrThresh = 0.0;

    std::string modelpath = ".\\yolov4-tiny.weights";
	std::string configpath = ".\\yolov4-tiny.cfg";
	std::string framework = "DarkNet";

    ObjDetection detectorObject = ObjDetection(0.2, 0.5, 0.00392, 
		true, 416, 416, cv::Scalar(0,0,0), modelpath, configpath, framework);

    std::vector<objectDetected> detection_list;
    cv::Point clickedPoint_pt = cv::Point(-1, -1);  

    namedWindow("Frame");
    
    while (capture.read(frame)) 
    {       
        // resize to (1280 x 720)
        resize(frame, frame, cv::Size(1280, 720));

        // detection runs every frame
        detection_list.clear();    
        isDetected = detectorObject.returnDetections(frame, detection_list);    
        //isDetected = false;

        if (!isTrackerInitialized_b) 
        {
            // Perform object detection to initialize the tracker            
            detection_list.clear();            

            // detection
            detectorObject.returnDetections(frame, detection_list);

            for(int i=0; i<detection_list.size(); i++) {
                Rect temp = detection_list[i].enclosingBox;
                rectangle(frame, temp, Scalar(0,255,255),2,8);
            }

            cv::setMouseCallback("Frame", onMouse, (void*)&clickedPoint_pt);

            // check for mouse click
            if(clickedPoint_pt.x > 0)
                isObjectSelected = returnObjectToTrack(detection_list, clickedPoint_pt, detectedBoundingBox);    
            
            if (isObjectSelected) {

                // Initialize Kalman filter with detected bounding box coordinates
                kalmanFilter.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);
                kalmanFilter.statePre.at<float>(0) = detectedBoundingBox.x + detectedBoundingBox.width / 2;
                kalmanFilter.statePre.at<float>(1) = detectedBoundingBox.y + detectedBoundingBox.height / 2;

                setIdentity(kalmanFilter.measurementMatrix);
                setIdentity(kalmanFilter.processNoiseCov, Scalar::all(1e-4));   
                setIdentity(kalmanFilter.measurementNoiseCov, Scalar::all(1e-1));
                setIdentity(kalmanFilter.errorCovPost, Scalar::all(1));
                
                // Initialize MOSSE tracker with the detected bounding box
                tracker = legacy::upgradeTrackingAPI(legacy::TrackerMOSSE::create());
                tracker->init(frame, detectedBoundingBox);

                //bool isMosseInit = mTracker.init(frame, detectedBoundingBox);

                //cv::putText(frame, "Mosse Init: " + to_string(isMosseInit), Point(1100, 50), 0, 0.75, Scalar(0,0,0),2,8);
                // kalman update is needed here ??
                
                // Switch to tracking mode              
                isTrackerInitialized_b = true;
                trackedFrameCount++;
                isObjectSelected = false;
            }
        } 

        else {

            // Update Kalman filter prediction
            prediction = kalmanFilter.predict();

            // check every 30th frame to use detections
            if(trackedFrameCount % 30 == 0)
                useDetections = true;
            
            if(useDetections && isDetected) {

                //useDetections = false;
                isObjectSelected = returnObjectToTrack(detection_list, trackedBoundingBox, detectedBoundingBox);
                //std::cout << "Obj Selected : " << isObjectSelected << std::endl; 

                if(isObjectSelected) {
                    tracker.release();
                    tracker = legacy::upgradeTrackingAPI(legacy::TrackerMOSSE::create());                    
                    tracker->init(frame, detectedBoundingBox);
                    
                    //trackedBoundingBox = detectedBoundingBox;
                    //tracker->update(frame, trackedBoundingBox);

                    //bool isMosseInitOrNot = mTracker.init(frame, detectedBoundingBox);
                    //mTracker.~MosseTracker();
                    //bool isMosseInitOrNot = mTracker.init(frame, detectedBoundingBox);
                    //mTracker.update(frame, detectedBoundingBox, psrThresh);
                }

                drawRect(frame, detectedBoundingBox, Scalar(255,0,0));
                trackedBoundingBox = detectedBoundingBox;
            }   
            
            // Update MOSSE tracker
            bool trackingSuccess = tracker->update(frame, trackedBoundingBox);

            //bool trackingSuccess = mTracker.update(frame, trackedBoundingBox, psrThresh);
            //cv::putText(frame, "PSR : " + to_string(psrThresh), Point(trackedBoundingBox.x, 
            //  trackedBoundingBox.y-20), 0, 0.75, Scalar(0,0,0),2,8);
            
            if (trackingSuccess) {

                circle(frame, Point(50,50), 10, Scalar(0,255,0), -1, 8);

                // Use MOSSE tracker output as measurement for Kalman filter
                measurement.at<float>(0) = trackedBoundingBox.x + trackedBoundingBox.width / 2;
                measurement.at<float>(1) = trackedBoundingBox.y + trackedBoundingBox.height / 2;
                
                // Update Kalman filter with the measurement
                Mat estimated = kalmanFilter.correct(measurement);
                
                // Update tracked bounding box with Kalman filter estimation
                trackedBoundingBox.x = estimated.at<float>(0) - trackedBoundingBox.width / 2;
                trackedBoundingBox.y = estimated.at<float>(1) - trackedBoundingBox.height / 2;

                cv::putText(frame, "[x,y] : " + to_string(trackedBoundingBox.x) + "," + 
                        to_string(trackedBoundingBox.y) ,Point(1000, 50), 0, 0.75, Scalar(0,255,255),2,8);
              
                // Draw tracked bounding box
                drawRect(frame, trackedBoundingBox, Scalar(0,255,0));

                trackingConf++;
                trackedBoundingBoxCopy = trackedBoundingBox;
                trackedFrameCount++;
            }

            else {

                trackingConf = trackingConf - 1;
                circle(frame, Point(50,50), 10, Scalar(0,0,255), -1, 8);

                if(trackingConf > 0) {

                    Point predicted = Point((int)prediction.at<float>(0),(int)prediction.at<float>(1));
                    cv::putText(frame, "[x,y] : " + to_string(predicted.x) + "," + 
                            to_string(predicted.y) ,Point(1000, 50), 0, 0.75, Scalar(0,255,255),2,8);

                    predictedBox = boxFromPtAndRect(predicted, trackedBoundingBoxCopy); 
                    drawRect(frame, predictedBox, Scalar(0,0,255));
                }

                else {
                    // If tracking fails, reinitialize 
                    isTrackerInitialized_b = false;
                    clickedPoint_pt.x = -1;
                    //trackedFrameCount = 0;
                }
            }
        }

        if(trackingConf > 100)
            trackingConf = 100;

        if(trackingConf < 0)
            trackingConf = 0;

        cv::putText(frame, "Conf : " + to_string(trackingConf), Point(1000, 80), 
                        0, 0.75, Scalar(0,255,255), 2, 8);
        

        //vidOut << frame;
        // Display frame
        imshow("Frame", frame);

        char key = waitKey(1);

        if(key == 'p')
            waitKey(0);

        if(key == 'r') {

            isTrackerInitialized_b = false;
            clickedPoint_pt.x = -1;
            trackingConf = 0;
            trackedFrameCount = 0;
            //mTracker.~MosseTracker();
        }
        
        // Exit loop if 'q' is pressed
        if (key == 'q')                
            break;

    }
}
