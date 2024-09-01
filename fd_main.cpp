#include <opencv2/opencv.hpp>

#include <vector>
#include <iostream>
#include <random>
#include <cmath>

//#include "lsd.h"
#include "featuredetection.h"

using namespace cv;
using namespace std;

// Custom comp
Scalar randomColor()
{
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(0, 255);
    return Scalar(dist(gen), dist(gen), dist(gen)); // 임의의 BGR 색상 생성
}

int main()
{

    // 이미지 파일 경로
    std::string home_path = getenv("HOME");
    cv::Mat raw_img = cv::imread(home_path + "/myStudyCode/regonSeg/imgdb/occupancy_grid.png", cv::IMREAD_GRAYSCALE);
    //cv::Mat raw_img = cv::imread(home_path +"/myStudyCode/regonSeg/imgdb/caffe_map.pgm", cv::IMREAD_GRAYSCALE);
    if (raw_img.empty())
    {
        std::cerr << "Error: Unable to open image file: " << std::endl;
        return -1;
    }

    std::vector<cv::Point> featurePoints;
    FeatureDetection fd(raw_img, featurePoints);
    cv::Mat img_lsd = fd.straightLineDetection();   

    fd.detectEndPoints(img_lsd, 12);
    std::vector<cv::Point> updata_featurePoints = fd.updateFeaturePoints();

    fd.imgShow(updata_featurePoints);   
    waitKey(0);

    return 0;
}