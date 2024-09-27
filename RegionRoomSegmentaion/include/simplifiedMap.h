#ifndef _SIMPLIFIEDMAP_H
#define _SIMPLIFIEDMAP_H

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <math.h>

// using namespace std;
// using namespace cv;

enum CATEGORYMAP
{
    BACKGROUND, // 배경
    DRIVING,    // 주행구간
    WALL,       // 벽 구간
    UNDRIVING   // 불확실한 주행구간
};

struct BOX
{
    int state = BACKGROUND;
    cv::Rect roi;
    //bool feature = false;
    double distance;
};

struct BOXINFO
{
    std::vector<std::pair<int, int>> dir;
    BOX info;
};

class SimplifyMap
{
public:
    SimplifyMap();
    ~SimplifyMap();

    
    cv::Mat runSimplify(cv::Mat img_gray);  
    cv::Mat makeImgGradient(cv::Mat img_gray); 
    std::vector<std::vector<BOX>> blockification(cv::Mat img_gray);
 
    double converterMean(cv::Mat img_box);
    int normalizeMat2Percent(cv::Mat m_arry);
    
    bool wallLineMap(cv::Mat img_box, int type, int threshold_pixel = 6);
    void checkBlockWall(std::vector<std::vector<BOX>> &info_box, cv::Mat img_gray, cv::Mat img_gradient); 
    void checkUndriving(std::vector<std::vector<BOXINFO>> &pixelBox);
    std::vector<std::vector<BOXINFO>> get8Neighbors(std::vector<std::vector<BOX>> info_box);
 
    cv::Mat makeSimplifiedImageColor(std::vector<std::vector<BOXINFO>> info, cv::Mat img_gray);
    cv::Mat makeSimplifiedImage(std::vector<std::vector<BOXINFO>> info, cv::Mat img_gray); 


private:
    // cv::Mat m_img_gray;
    // cv::Mat m_img_gradient;
    //cv::Mat m_img_color;
 

};

#endif // _PLINKAGE_SUPERPIXEL_
