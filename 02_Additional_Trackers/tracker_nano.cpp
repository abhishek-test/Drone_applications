#include <opencv2/opencv.hpp>
#include "opencv2/video/tracking.hpp"

using namespace cv;
using namespace std;

int main() {

    cout << "Entered main() " << endl;

    cv::VideoCapture cap("C:\\Abhishek_Data\\My_Data\\Datasets\\videos\\drone_traffic.mp4");

    if (!cap.isOpened()) {
        std::cerr << "Error opening the camera." << std::endl;
        return -1;
    }

    cv::namedWindow("Webcam", cv::WINDOW_NORMAL);

    TrackerNano::Params params = TrackerNano::Params();
    params.backbone = "C:\\Users\\abhishek.sri\\Downloads\\nanotrack_backbone_sim_v2.onnx";
    params.neckhead = "C:\\Users\\abhishek.sri\\Downloads\\nanotrack_head_sim_v2.onnx";
    params.backend = 0;
    params.target = 0;

    cv::Ptr<TrackerNano> tracker = cv::TrackerNano::create(params);
    
    cv::Mat frame;
    cap >> frame;

    resize(frame, frame, Size(1280/1, 720/1));

    Rect roi = selectROI("Webcam", frame);
    bool isInit = false;

    int t1, t2;
    double tickfreq = cv::getTickFrequency();

    

    while (true) {

        t1 = t2 = 0;
        
        cap >> frame;
        resize(frame, frame, Size(1280/1, 720/1));

        if(!isInit) {
            tracker->init(frame, roi);
            isInit = true;
        }

        else {

            t1 = cv::getTickCount();

            tracker->update(frame, roi);
            float conf = tracker->getTrackingScore();

            t2 = cv::getTickCount();
            double fps = tickfreq/(t2-t1);

            rectangle(frame, roi, Scalar(0,255,0), 2, 8);
            putText(frame, to_string((int)(conf*100)), Point2d(roi.x, roi.y-5), FONT_HERSHEY_COMPLEX, 0.3, Scalar(0,0,0));

            rectangle(frame, Rect(100, 20, 85, 20), Scalar(0,0,0), -1, 8);
            putText(frame, "FPS: " + to_string((int)(fps)), Point(100, 33), FONT_HERSHEY_COMPLEX, 0.3, Scalar(0,255,255));
        }

        cv::imshow("Webcam", frame);

        if (cv::waitKey(1) == 'q') {
            break;
        }

        if (cv::waitKey(1) == 'r') {
            roi = selectROI("Webcam", frame);
            tracker->init(frame, roi);
        }

    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
