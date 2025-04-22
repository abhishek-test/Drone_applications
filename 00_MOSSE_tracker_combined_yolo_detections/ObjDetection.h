#ifndef _OBJECT_DETECT_H_
#define _OBJECT_DETECT_H_

#include <fstream>
#include <sstream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

typedef struct {
	cv::Rect enclosingBox;
	float detConfidence;
	cv::String classIdentity;
}objectDetected;

class ObjDetection {

public:
	float confThreshold;
	float nmsThreshold;
	float scale;	
	bool swapRB;
	int inpWidth;
	int inpHeight;

	cv::Scalar mean;
	cv::dnn::Net net;

	std::string modelPath;
	std::string configPath;
	std::string framework;
	std::vector<cv::String> outNames;

public:

	ObjDetection(float confThreshold_p, float nmsThreshold_p, float scale_p, 
		float swapRB_p, int inpWidth_p, int inpHeight_p, cv::Scalar mean_p, 
		std::string modelPath_p, std::string configPath_p, std::string frameWork_p);

	void preprocess(const cv::Mat& frame, cv::dnn::Net& net, cv::Size inpSize, 
		float scale, const cv::Scalar& mean, bool swapRB);

	void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& out, cv::dnn::Net& net, 
		int backend, std::vector<objectDetected>& detection_boxes);

	bool returnDetections(cv::Mat &inputImg_mt, std::vector<objectDetected> &detection_boxes);
};

#endif