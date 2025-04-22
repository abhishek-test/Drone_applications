
#include <iostream>
#include <cmath>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace cv::dnn;

static
int run(int argc, char** argv)
{
    std::cout << "Entered main ()" << std::endl;

    cv::VideoCapture cap("C:\\Abhishek_Data\\My_Data\\Datasets\\videos\\drone_traffic.mp4");

    if (!cap.isOpened()) {
        std::cerr << "Error opening the camera." << std::endl;
        return -1;
    }

    //cv::namedWindow("Webcam", cv::WINDOW_NORMAL);

    TrackerVit::Params params;
    params.net = "C:\\Users\\abhishek.sri\\Downloads\\object_tracking_vittrack.onnx";
    params.backend = 0;
    params.target = 0;
    cv::Ptr<TrackerVit> tracker = TrackerVit::create(params);
    

    const std::string winName = "vitTracker";
    namedWindow(winName, WINDOW_AUTOSIZE);

    // Read the first image.
    Mat image;
    cap >> image;
    resize(image, image, Size(1280, 720));
    
    if (image.empty())
    {
        std::cerr << "Can't capture frame!" << std::endl;
        return 2;
    }

    Mat image_select = image.clone();
    putText(image_select, "Select initial bounding box you want to track.", Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
    putText(image_select, "And Press the ENTER key.", Point(0, 35), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

    Rect selectRect = selectROI(winName, image_select);
    std::cout << "ROI=" << selectRect << std::endl;

    tracker->init(image, selectRect);

    TickMeter tickMeter;

    for (int count = 0; ; ++count)
    {
        cap >> image;
        resize(image, image, Size(1280, 720));

        if (image.empty())
        {
            std::cerr << "Can't capture frame " << count << ". End of video stream?" << std::endl;
            break;
        }

        Rect rect;

        tickMeter.start();
        bool ok = tracker->update(image, rect);
        tickMeter.stop();

        float score = tracker->getTrackingScore();

        Mat render_image = image.clone();

        if (ok)
        {
            rectangle(render_image, rect, Scalar(0, 255, 0), 2);

            std::string timeLabel = format("FPS: %2d", (int)(1000.0/(1.0*tickMeter.getTimeMilli())));
            std::string scoreLabel = format("Score: %f", score);
            putText(render_image, timeLabel, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
            putText(render_image, scoreLabel, Point(0, 35), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        }

        imshow(winName, render_image);

        tickMeter.reset();

        int c = waitKey(1);
        if (c == 'q' /*ESC*/)
            break;
    }

    std::cout << "Exit" << std::endl;
    return 0;
}


int main(int argc, char **argv)
{
    try
    {
        return run(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::cerr << "FATAL: C++ exception: " << e.what() << std::endl;
        return 1;
    }
}