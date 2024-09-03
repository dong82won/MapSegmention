#ifndef FEATUREDETECTION_H
#define FEATUREDETECTION_H

#include <opencv2/opencv.hpp>

#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <vector>

#include "lsd.h"

class FeatureDetection
{

    private:
        cv::Mat img_raw_;     
        cv::Mat img_lsd_;   
        // cv::Mat img_lsd_;
        std::vector<cv::Point> featurePoints_;
        std::vector<cv::Point> upddateFeaturePoints_;

        void mergeClosePoints(std::vector<cv::Point> &points, int distanceThreshold);
        cv::Point calculateMinimumPointAround(cv::Point featurePoint);

    public:
        /* data */
        FeatureDetection();
        FeatureDetection(const cv::Mat& img_raw);        
        ~FeatureDetection();
        
        
        void straightLineDetection();
        void detectEndPoints(int distanceThreshold);

        std::vector<cv::Point> getUpdateFeaturePoints();
        cv::Mat getDetectLine();
};


#endif // FEATUREDETECTION_H
