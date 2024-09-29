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
    cv::Mat raw_img = cv::imread(home_path + "/myStudyCode/MapSegmention/imgdb/occupancy_grid.png", cv::IMREAD_GRAYSCALE);    
    //cv::Mat raw_img = cv::imread(home_path + "/myStudyCode/MapSegmention/imgdb/caffe_map.pgm", cv::IMREAD_GRAYSCALE);

    if (raw_img.empty())
    {
        std::cerr << "Error: Unable to open image file: " << std::endl;
        return -1;
    }
    imshow("raw_img", raw_img);


    ROOMSEG rs;

    rs.extractWallElements(raw_img);
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
    
    rs.makeRotatedImage(raw_img);
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

    //cv::Mat free_space = rs.getImageFreeSpace();  


    rs.classificationRegon();
    std::vector<cv::Rect> region = rs.getRegionBox();    
     
    //rs.segmentationRegion();
    rs.makeRegionContour();


    // cv::imshow("img_grid2", img_grid); 

    // cv::Mat test = cv::Mat::zeros(img_rotated.size(), CV_8UC3);

    // for (const auto& box : region)
    // {
    //     //cv::rectangle(sug, box ,cv::Scalar(255), -1);

    //     cv::Scalar color = randomColor();

    //     // 관심 영역만 추출
    //     cv::Mat roiImage = img_grid(box);


    //     // 윤곽선 검출
    //     std::vector<std::vector<cv::Point>> contours;
    //     cv::findContours(roiImage, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        
    //     // 원본 이미지에 윤곽선 그리기
    //     for (size_t i = 0; i < contours.size(); i++) 
    //     {
    //         // 윤곽선을 원본 이미지에 그립니다.
    //         cv::drawContours(test, contours, static_cast<int>(i), color, 
    //                          1, cv::LINE_4, cv::noArray(), 0, box.tl());
    //     }

    // }

    
    // cv::imshow("test", test); 

    cv::imshow("color_img_grid", color_img_grid); 
    cv::waitKey(0);

    return 0;
}

