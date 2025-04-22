#include "opencv2/opencv.hpp"
//#include "opencv2/tracking/tracking_legacy.hpp"


using namespace cv;

class MosseTracker {

private:
    Point2d center; //center of the bounding box
    Size size;      //size of the bounding box
    Mat hanWin;
    Mat G;          //goal
    Mat H, A, B;    //state
    const double eps=0.00001;      // for normalization
    const double rate=0.2;         // learning rate
    const double psrThreshold=5.7; // no detection, if PSR is smaller than this

protected:
    Mat divDFTs( const Mat &src1, const Mat &src2 );
    void preProcess( Mat &window );
    double correlate( const Mat &image_sub, Point &delta_xy );
    Mat randWarp( const Mat& a );

 public:   
    bool init( const Mat& image, const Rect2d& boundingBox );
    bool update( const Mat& image, Rect& boundingBox, double &psrVal );
    ~MosseTracker();

};