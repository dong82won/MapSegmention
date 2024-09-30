#include <opencv2/opencv.hpp>

#include <vector>
#include <iostream>
#include <random>
#include <cmath>

//#include "utility.h"
#include "roomSeg.h"

using namespace cv;
using namespace std;


int main()
{

    //1. 이미지 입력 및 회전 -------------------------------------------------------
    std::string home_path = getenv("HOME");
    // std::cout << home_path << std::endl;

    // 이미지 파일 경로
    cv::Mat img_raw = cv::imread(home_path + "/myWorkCode/MapSegmention/imgdb/occupancy_grid.png", cv::IMREAD_GRAYSCALE);    
    //cv::Mat img_raw = cv::imread(home_path + "/myStudyCode/MapSegmention/imgdb/caffe_map.pgm", cv::IMREAD_GRAYSCALE);

    if (img_raw.empty())
    {
        std::cerr << "Error: Unable to open image file: " << std::endl;
        return -1;
    }
    imshow("img_raw", img_raw);


    ROOMSEG rs;
    rs.extractWallElements(img_raw);
        
    rs.makeRotatedImage();

    cv::Mat img_wall = rs.getImgWall();
    imshow("occupancyMap", img_wall);    

    cv::Mat img_rotated = rs.getImgWallRotated();
    imshow("img_rotated", img_rotated);    

    //이미지 입력 및 회전 -------------------------------------------------------
    rs.makeGridSnappingContours();   
    
    cv::Mat img_grid = rs.getImgGridSnapping();    
    imshow("img_grid", img_grid);
    cv::Mat img_grid_skeletion = rs.getImgGridSnapping2();    
    imshow("img_grid_skeletion", img_grid_skeletion);
    
    rs.extractFeaturePts();
    std::vector<cv::Point> featurePts = rs.getFeaturePts();

    cv::Mat color_img_grid;
    cv::cvtColor(img_grid, color_img_grid, COLOR_GRAY2BGR);    
    for (const auto &pt : featurePts)
    {
        cv::circle(color_img_grid, pt, 3, cv::Scalar(0, 255, 0), -1);
    }
    
    rs.makeRotatedImage(img_raw);    
    cv::Mat img_raw_rotated =rs.getRotatedImage();
    cv::imshow("img_raw_rotated", img_raw_rotated);

    rs.extracTrajectorPts();
    std::vector<cv::Point> trajectoryPts = rs.getTrajectoryPts();    
    //cv::imshow("color_img_raw_rotated", color_img_raw_rotated);

    rs.extractVirtualLine();
    std::vector<LINEINFO> vitual_lines = rs.getVirtualLines();

    std::cout << "vitual_lines.size(): " << vitual_lines.size() << std::endl;    
    for (const auto &line : vitual_lines)
    {
        std::cout << "Line: ("
                << line.virtual_wll.first.x << ", " << line.virtual_wll.first.y << ") to ("
                << line.virtual_wll.second.x << ", " << line.virtual_wll.second.y << ") - Distance: "
                << line.distance << std::endl;
        cv::line(color_img_grid, line.virtual_wll.first, line.virtual_wll.second, CV_RGB(255, 0, 0), 3);
    }
    
    rs.classificationRegon();    
    std::vector<cv::Rect> region = rs.getRegionBox();        



    // rs.makeRotatedAngle();
    // cv::Mat img_seg = rs.getSegImage();
    // cv::imshow("img_seg", img_seg); 
    
    #다시 회전 원복 후 라벨링 알고리즘 적용해아 함으로 순서 변경 필요함!!
    rs.makeRegionContour();
    cv::Mat img_label = rs.getLabelImage();


//     cv::Mat img_color_raw;
//     cvtColor(img_raw, img_color_raw, COLOR_GRAY2BGR);

//  // Alpha blending을 위한 변수 설정 (투명도)
//     double alpha = 0.5;  // 첫 번째 이미지의 가중치
//     double beta = 1.0 - alpha;  // 두 번째 이미지의 가중치

//     cv::Mat blended;
//     // 두 이미지를 중첩합니다
//     cv::addWeighted(img_color_raw, alpha, img_seg, beta, 0.0, blended);

//     // 결과 이미지를 출력합니다
//     cv::imshow("Blended Image", blended);


    


    cv::imshow("color_img_grid", color_img_grid); 
    cv::waitKey(0);

    return 0;
}

