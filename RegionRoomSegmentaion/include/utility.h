#ifndef _UTILLITY_H
#define _UTILLITY_H

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
//#include <cmath>
#include <random>


// // Custom comp
cv::Scalar randomColor();
std::vector<cv::Point> changeMatoPoint(cv::Mat &image);
double calculateDistance(const cv::Point &p1, const cv::Point &p2);
std::vector<cv::Point> sortPoints(const std::vector<cv::Point> &points);
cv::Mat extractMat2Mat(cv::Mat &image, uchar pixel_value=128);

void removeDuplicatePoints(std::vector<cv::Point> &points);
void mergeClosePoints(std::vector<cv::Point> &points, int distanceThreshold);
void drawingSetpRectangle(cv::Mat &image, cv::Point circlesCenters, int radius);
std::vector<cv::Point> edgePointsInCircle(const cv::Point &center, int radius);
void drawingOutLineCircule(const cv::Mat &image,cv::Point circlesCenters, int radius);
#endif 
