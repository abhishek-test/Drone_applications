#ifndef _UTILITY_H_
#define _UTILITY_H_

#include "opencv2/opencv.hpp"
#include "ObjDetection.h"

void onMouse(int evt, int x, int y, int flags, void* param);
float eucledianDistance(cv::Point &firstPoint_pt, cv::Point &secondPoint_pt);
cv::Point rect2Point(cv::Rect &box);
cv::Rect getBoxFromCenterAndRect(cv::Point &centre, cv::Rect &rect);

bool returnObjectToTrack(std::vector<objectDetected> detections_v,
	cv::Point& clickedPointOnStream_pt, cv::Rect &obj2Track);

bool returnObjectToTrack(std::vector<objectDetected> detections_v,
	cv::Rect& box, cv::Rect &obj2Track);


#endif


