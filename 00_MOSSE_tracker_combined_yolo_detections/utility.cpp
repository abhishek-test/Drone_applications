#include "utility.h"


void onMouse(int evt, int x, int y, int flags, void* param) {
    if (evt == cv::EVENT_LBUTTONDOWN) {
        cv::Point* ptPtr = (cv::Point*)param;
        ptPtr->x = x;
        ptPtr->y = y;
    }
}


float eucledianDistance(cv::Point &firstPoint_pt, cv::Point &secondPoint_pt) {

    float deltaXSquared_f = powf((firstPoint_pt.x - secondPoint_pt.x), 2.0);
    float deltaYSquared_f = powf((firstPoint_pt.y - secondPoint_pt.y), 2.0);
    float deltaSquared = deltaXSquared_f - deltaYSquared_f;

    if (deltaSquared < 0)
        deltaSquared = -deltaSquared;

    return powf(deltaSquared, 0.5);
}


cv::Point rect2Point(cv::Rect &box) {
    return cv::Point(box.x + (int)(box.width / 2), box.y + (int)(box.height / 2));
}

float iou(cv::Rect &r1, cv::Rect &r2) {

    cv::Rect rectIntersection = (r1 & r2);
    cv::Rect rectUnion = (r1 | r2);
    return (rectIntersection.area()/(1.0*rectUnion.area()));
}


bool returnObjectToTrack(std::vector<objectDetected> detections_list,
    cv::Point& clickedPointOnStream_pt, cv::Rect& obj2Track) {

    if (detections_list.size() == 0)
        return false;

    float maxIOU = 0.1;
    float iouVal;
    int maxIdx = -1;
    cv::Rect boxSelected = cv::Rect(clickedPointOnStream_pt.x-50, clickedPointOnStream_pt.y-50, 100, 100);

    for (int i = 0; i < detections_list.size(); i++) {

        iouVal = iou(detections_list[i].enclosingBox, boxSelected);

        if (iouVal > maxIOU) {
            maxIOU = iouVal;
            maxIdx = i;
        }
    }    

    if (maxIdx == -1) {
        //obj2Track = boxSelected;
        return false;
    }

    else {
        obj2Track = detections_list[maxIdx].enclosingBox;
        return true;
    }

}

bool returnObjectToTrack(std::vector<objectDetected> detections_list,
	cv::Rect& boxSelected, cv::Rect &obj2Track) {

    if (detections_list.size() == 0)
        return false;

    float maxIOU = 0.1;
    float iouVal;
    int maxIdx = -1;

    for (int i = 0; i < detections_list.size(); i++) {

        iouVal = iou(detections_list[i].enclosingBox, boxSelected);

        if (iouVal > maxIOU) {
            maxIOU = iouVal;
            maxIdx = i;
        }
    }    

    if (maxIdx == -1) {
        return false;
    }

    else {
        obj2Track = detections_list[maxIdx].enclosingBox;
        return true;
    }
}


cv::Rect getBoxFromCenterAndRect(cv::Point& centre, cv::Rect& rect) {
    cv::Rect r = cv::Rect();

    r.x = centre.x - int(rect.width / 2);
    r.y = centre.y - int(rect.height / 2);
    r.width = rect.width;
    r.height = rect.height;

    return r;
}

/*
double getSimilarity(cv::Mat& A, cv::Mat& B) {

    cv::Mat firstImg_mt, secondImg_mt;

    cv::cvtColor(A, firstImg_mt, cv::COLOR_BGR2GRAY);
    cv::cvtColor(B, secondImg_mt, cv::COLOR_BGR2GRAY);

    if (firstImg_mt.rows > 0 && firstImg_mt.rows == secondImg_mt.rows && firstImg_mt.cols > 0 && firstImg_mt.cols == secondImg_mt.cols) {

        double errorL2 = norm(firstImg_mt, secondImg_mt, );
        double similarity = errorL2 / (double)(A.rows * A.cols);
        return similarity;
    }
    else {
        //Images have a different size
        return 100000000.0;  // Return a bad value
    }
}
*/