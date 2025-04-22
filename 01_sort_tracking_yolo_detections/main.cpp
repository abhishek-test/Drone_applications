
#include <iostream>
#include <fstream>
#include <iomanip> 
#include <io.h> 
#include <set>

#include "Hungarian.h"
#include "KalmanTracker.h"
#include "ObjDetection.h"

#include "opencv2/opencv.hpp"
#include "opencv2/video/tracking.hpp"

using namespace std;
using namespace cv;

typedef struct TrackingBox
{
	int frame;
	int id;
	Rect_<float> box;
}TrackingBox;


// Computes IOU between two bounding boxes
double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
	float in = (bb_test & bb_gt).area();
	float un = bb_test.area() + bb_gt.area() - in;

	if (un < DBL_EPSILON)
		return 0;

	return (double)(in / un);
}

// global variables for counting
#define CNUM 20
int total_frames = 0;
double total_time = 0.0;

int main()
{
	//VideoCapture capture("C:\\Abhishek_Data\\Tunga_Project\\videos\\Car_Crossroad.mp4");
	VideoCapture capture("..\\tracker\\video.mp4");
	//VideoCapture capture("rtsp://192.168.0.101:8080/h264_pcm.sdp");

	if(!capture.isOpened()) {
		cout << " !! Error opening video src !! " << std::endl;
		return -1;
	}

	// detction variables
	std::string modelpath = "..\\tracker\\yolov4-tiny.weights";
	std::string configpath = "..\\tracker\\yolov4-tiny.cfg";
	std::string framework = "DarkNet";

    ObjDetection detectorObject = ObjDetection(0.2, 0.5, 0.00392, 
		true, 416, 416, cv::Scalar(0,0,0), modelpath, configpath, framework);

    std::vector<objectDetected> detection_list;
	vector<TrackingBox> detFrameData;
	Mat frame;

	/*
	while (capture.read(frame)) 
    {       
        // detection runs every frame
        detection_list.clear();    
        bool isDetected = detectorObject.returnDetections(frame, detection_list);

		for (int i=0; i<detection_list.size(); i++)
		{
			Rect temp = detection_list[i].enclosingBox;
			rectangle(frame, temp, Scalar(0,255,255),1,8);
			cv::putText(frame, detection_list[i].classIdentity, Point(temp.x, temp.y-5), 0, 0.75, Scalar(0,0,0),2,8);
			cv::putText(frame, to_string((int)(detection_list[i].detConfidence * 100)), Point(temp.x, temp.y + temp.height + 5), 0, 0.75, Scalar(0,0,0),2,8);
		}

		imshow("Detections", frame);
		char key = waitKey(1);

		if(key == 'q')
			break;
	}
	*/
    
	// randomly generate colors, only for display
	RNG rng(0xFFFFFFFF);
	Scalar_<int> randColor[CNUM];
	for (int i = 0; i < CNUM; i++)
		rng.fill(randColor[i], RNG::UNIFORM, 0, 256);

	// update across frames
	int frame_count = 0;
	int max_age = 100;  
	int min_hits = 3;
	double iouThreshold = 0.3;
	vector<KalmanTracker> trackers;
	// tracking id relies on this, so we have to reset it in each seq.
	KalmanTracker::kf_count = 0; 

	// variables used in the for-loop
	vector<Rect_<float>> predictedBoxes;
	vector<vector<double>> iouMatrix;
	vector<int> assignment;
	set<int> unmatchedDetections;
	set<int> unmatchedTrajectories;
	set<int> allItems;
	set<int> matchedItems;
	vector<cv::Point> matchedPairs;
	vector<TrackingBox> frameTrackingResult;
	unsigned int trkNum = 0;
	unsigned int detNum = 0;

	double cycle_time = 0.0;
	int64 start_time = 0;

	// main loop
	while(1)
	{
		total_frames++;
		frame_count++;				
		start_time = getTickCount();

		detFrameData.clear();
		detection_list.clear();

		capture >> frame;
		bool isDetected = detectorObject.returnDetections(frame, detection_list);

		for(int i=0; i<detection_list.size(); i++) {
			TrackingBox temp;
			
			temp.box = Rect2f(detection_list[i].enclosingBox.x, detection_list[i].enclosingBox.y, 
								detection_list[i].enclosingBox.width, detection_list[i].enclosingBox.height);

			//temp.id = -1;
			detFrameData.push_back(temp);
		}

		// the first frame met
		if (trackers.size() == 0) {

			// initialize kalman trackers using first detections.
			for (unsigned int i = 0; i < detFrameData.size(); i++) {
				KalmanTracker trk = KalmanTracker(detFrameData[i].box);
				trackers.push_back(trk);
			}

			// output the first frame detections
			for (unsigned int id = 0; id < detFrameData.size(); id++) {
				TrackingBox tb = detFrameData[id];
			}

			continue;
		}

		// get predicted locations from existing trackers.
		predictedBoxes.clear();

		for (auto it = trackers.begin(); it != trackers.end();) {
			Rect_<float> pBox = (*it).predict();

			if (pBox.x >= 0 && pBox.y >= 0) {
				predictedBoxes.push_back(pBox);
				it++;
			}
			else {
				it = trackers.erase(it);
			}
		}

		// associate detected bounding boxes to tracked object 
		trkNum = predictedBoxes.size();
		detNum = detFrameData.size();

		iouMatrix.clear();
		iouMatrix.resize(trkNum, vector<double>(detNum, 0));

		// compute iou matrix as a distance matrix
		for (unsigned int i = 0; i < trkNum; i++) {
			for (unsigned int j = 0; j < detNum; j++) {
				// use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
				iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detFrameData[j].box);
			}
		}

		// solve the assignment problem using hungarian algorithm.
		// the resulting assignment is [track(prediction) : detection], with len=preNum
		HungarianAlgorithm HungAlgo;
		assignment.clear();
		HungAlgo.Solve(iouMatrix, assignment);

		// find matches, unmatched_detections and unmatched_predictions
		unmatchedTrajectories.clear();
		unmatchedDetections.clear();
		allItems.clear();
		matchedItems.clear();

		//	there are unmatched detections
		if (detNum > trkNum) {
			for (unsigned int n = 0; n < detNum; n++)
				allItems.insert(n);

			for (unsigned int i = 0; i < trkNum; ++i)
				matchedItems.insert(assignment[i]);

			set_difference(allItems.begin(), allItems.end(),
				matchedItems.begin(), matchedItems.end(),
				insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
		}

		// there are unmatched trajectory/predictions
		else if (detNum < trkNum) {
			for (unsigned int i = 0; i < trkNum; ++i)
				// unassigned label will be set as -1 in the assignment algorithm
				if (assignment[i] == -1) 
					unmatchedTrajectories.insert(i);
		}

		else
			;

		// filter out matched with low IOU
		matchedPairs.clear();
		for (unsigned int i = 0; i < trkNum; ++i) {
			if (assignment[i] == -1) // pass over invalid values
				continue;
			if (1 - iouMatrix[i][assignment[i]] < iouThreshold) {
				unmatchedTrajectories.insert(i);
				unmatchedDetections.insert(assignment[i]);
			}
			else
				matchedPairs.push_back(cv::Point(i, assignment[i]));
		}

		// update matched trackers with assigned detections.
		// each prediction is corresponding to a tracker
		int detIdx, trkIdx;
		for (unsigned int i = 0; i < matchedPairs.size(); i++) {
			trkIdx = matchedPairs[i].x;
			detIdx = matchedPairs[i].y;
			trackers[trkIdx].update(detFrameData[detIdx].box);
		}

		// create and initialize new trackers for unmatched detections
		for (auto umd : unmatchedDetections) {
			KalmanTracker tracker = KalmanTracker(detFrameData[umd].box);
			trackers.push_back(tracker);
		}

		// get trackers' output
		frameTrackingResult.clear();
		for (auto it = trackers.begin(); it != trackers.end();) {
			if (((*it).m_time_since_update < 1) &&
				((*it).m_hit_streak >= min_hits || frame_count <= min_hits)) {
				TrackingBox res;
				res.box = (*it).get_state();
				res.id = (*it).m_id + 1;
				res.frame = frame_count;
				frameTrackingResult.push_back(res);
				it++;
			}

			else
				it++;

			if (it != trackers.end() && (*it).m_time_since_update > max_age)
				it = trackers.erase(it);
		}

		cycle_time = (double)(getTickCount() - start_time);
		total_time += cycle_time / getTickFrequency();
		
		for (auto tb : frameTrackingResult) {
			cv::rectangle(frame, tb.box, randColor[tb.id % CNUM], 2, 8, 0); 
			cv::putText(frame, "Id:" + to_string(tb.id), cv::Point(tb.box.x, tb.box.y-5), 
							FONT_HERSHEY_COMPLEX_SMALL, 0.7, cv::Scalar(0,0,0), 1,8);
		}

		imshow("Tracking", frame);
		
		char key = waitKey(1);

		if(key == 'q')
			break;
		
	}
}
