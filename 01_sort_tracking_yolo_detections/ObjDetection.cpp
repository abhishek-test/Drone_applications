#include "ObjDetection.h"

std::string names_array[80] = {"person", "bicycle", "car", "motorbike", "aeroplane", "bus", 
                                "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", 
                                "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                                "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", 
                                "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", 
                                "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
                                "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", 
                                "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", 
                                "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table", 
                                "toilet", "TV monitor", "laptop", "mouse", "remote", "keyboard", "cell phone", 
                                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", 
                                "scissors", "teddy bear", "hair drier", "toothbrush"};

ObjDetection::ObjDetection(float confThreshold_p, float nmsThreshold_p, float scale_p, 
	float swapRB_p,	int inpWidth_p, int inpHeight_p, cv::Scalar mean_p, 
	std::string modelPath_p, std::string configPath_p, std::string frameWork_p) {

	confThreshold = confThreshold_p;
	nmsThreshold = nmsThreshold_p;
	scale = scale_p;
	swapRB = swapRB_p;
	inpWidth = inpWidth_p;
	inpHeight = inpHeight_p;
	mean = mean_p;	

	modelPath = modelPath_p;
	configPath = configPath_p;
	framework = frameWork_p;

	net = cv::dnn::readNet(modelPath, configPath, framework);   
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	outNames = net.getUnconnectedOutLayersNames();    
}


void ObjDetection::preprocess(const cv::Mat& frame, cv::dnn::Net& net,
	cv::Size inpSize, float scale, const cv::Scalar& mean, bool swapRB) {

	static cv::Mat blob;
	// Create a 4D blob from a frame.
	if (inpSize.width <= 0) inpSize.width = frame.cols;
	if (inpSize.height <= 0) inpSize.height = frame.rows;
	cv::dnn::blobFromImage(frame, blob, 1.0, inpSize, cv::Scalar(), swapRB, false, CV_8U);

	// Run a model.
	net.setInput(blob, "", scale, mean);
	if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
	{
		resize(frame, frame, inpSize);
		cv::Mat imInfo = (cv::Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
		net.setInput(imInfo, "im_info");
	}
}


void ObjDetection::postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, 
    cv::dnn::Net& net, int backend, std::vector<objectDetected>& detection_boxes) {

    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    if (outLayerType == "DetectionOutput")
    {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        CV_Assert(outs.size() > 0);
        for (size_t k = 0; k < outs.size(); k++)
        {
            float* data = (float*)outs[k].data;
            for (size_t i = 0; i < outs[k].total(); i += 7)
            {
                float confidence = data[i + 2];
                if (confidence > confThreshold)
                {
                    int left = (int)data[i + 3];
                    int top = (int)data[i + 4];
                    int right = (int)data[i + 5];
                    int bottom = (int)data[i + 6];
                    int width = right - left + 1;
                    int height = bottom - top + 1;

                    if (width <= 2 || height <= 2)
                    {
                        left = (int)(data[i + 3] * frame.cols);
                        top = (int)(data[i + 4] * frame.rows);
                        right = (int)(data[i + 5] * frame.cols);
                        bottom = (int)(data[i + 6] * frame.rows);
                        width = right - left + 1;
                        height = bottom - top + 1;
                    }

                    classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
                    boxes.push_back(cv::Rect(left, top, width, height));
                    confidences.push_back(confidence);
                }
            }
        }
    }

    else // (outLayerType == "Region")
    {
        for (size_t i = 0; i < outs.size(); ++i)
        {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]

            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                cv::Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold)
                {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }
    }

    // NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends we need NMS in sample
    // or NMS is required if number of outputs > 1
    if (outLayers.size() > 1 || (outLayerType == "Region" && backend != cv::dnn::DNN_BACKEND_OPENCV))
    {
        std::map<int, std::vector<size_t> > class2indices;
        for (size_t i = 0; i < classIds.size(); i++)
        {
            if (confidences[i] >= confThreshold)
            {
                class2indices[classIds[i]].push_back(i);
            }
        }
        std::vector<cv::Rect> nmsBoxes;
        std::vector<float> nmsConfidences;
        std::vector<int> nmsClassIds;
        for (std::map<int, std::vector<size_t> >::iterator it = class2indices.begin(); it != class2indices.end(); ++it)
        {
            std::vector<cv::Rect> localBoxes;
            std::vector<float> localConfidences;
            std::vector<size_t> classIndices = it->second;
            for (size_t i = 0; i < classIndices.size(); i++)
            {
                localBoxes.push_back(boxes[classIndices[i]]);
                localConfidences.push_back(confidences[classIndices[i]]);
            }
            std::vector<int> nmsIndices;
            cv::dnn::NMSBoxes(localBoxes, localConfidences, confThreshold, nmsThreshold, nmsIndices);
            for (size_t i = 0; i < nmsIndices.size(); i++)
            {
                size_t idx = nmsIndices[i];
                nmsBoxes.push_back(localBoxes[idx]);
                nmsConfidences.push_back(localConfidences[idx]);
                nmsClassIds.push_back(it->first);
            }
        }
        boxes = nmsBoxes;
        classIds = nmsClassIds;
        confidences = nmsConfidences;
    }

    for (size_t idx = 0; idx < boxes.size(); ++idx)
    {
        objectDetected temp;
        cv::Rect box = boxes[idx];        
        
        temp.enclosingBox = box;
        temp.classIdentity = names_array[classIds[idx]]; 
        temp.detConfidence = confidences[idx];

        detection_boxes.push_back(temp);
    }
}


bool ObjDetection::returnDetections(cv::Mat& inputImg_mt, 
    std::vector<objectDetected>& detection_boxes) {

    preprocess(inputImg_mt, net, cv::Size(inpWidth, inpHeight), scale, mean, swapRB);
	std::vector<cv::Mat> outs;
	net.forward(outs, outNames);
		
	postprocess(inputImg_mt, outs, net, 0, detection_boxes);

    if (detection_boxes.size() != 0)
        return true;
    else
        return false;
}
	
	



