#ifndef FEATUREDETECTION_H
#define FEATUREDETECTION_H

#include <opencv2/opencv.hpp>

#include <vector>
#include <iostream>
#include <random>
#include <cmath>


#include "lsd.h"

class FeatureDetection
{

    private:
        cv::Mat img_raw_;     
        cv::Mat img_lsd_;   
        // cv::Mat img_lsd_;
        // std::vector<cv::Point> &featurePoints_;

    public:
        /* data */
        FeatureDetection();
        FeatureDetection(const cv::Mat& img_raw);        
        

        // void mergeClosePoints(std::vector<cv::Point> &points, int distanceThreshold);
        // cv::Point calculateMinimumPointAround(cv::Point featurePoint);


        // FeatureDetection();
        // FeatureDetection(const cv::Mat &img, std::vector<cv::Point> &featurePoints);
        // //FeatureDetection(const cv::Mat &img);
        // ~FeatureDetection();

        // void imgShow(std::vector<cv::Point> &updateFeaturePoint);    
        
        // void straightLineDetection();

        // void detectEndPoints(const cv::Mat &imgLine, int distanceThreshold);
        // std::vector<cv::Point> updateFeaturePoints();
};


#endif // FEATUREDETECTION_H
