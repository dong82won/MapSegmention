#ifndef _ROOMSEG_H
#define _ROOMSEG_H

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "utility.h"
#include <queue>

using namespace cv;
using namespace std;



struct LINEINFO
{
    std::pair<cv::Point, cv::Point> virtual_wll;

    double distance;

    // 두 LINEINFO가 동일한지 비교하는 연산자
    bool operator==(const LINEINFO &other) const
    {
        // 비교: 거리와 두 점의 위치가 동일한지 체크
        return (distance == other.distance) &&
          ((virtual_wll.first.x == other.virtual_wll.first.x && virtual_wll.first.y == other.virtual_wll.first.y &&
            virtual_wll.second.x == other.virtual_wll.second.x && virtual_wll.second.y == other.virtual_wll.second.y) ||
          (virtual_wll.first.x == other.virtual_wll.second.x && virtual_wll.first.y == other.virtual_wll.second.y &&
            virtual_wll.second.x == other.virtual_wll.first.x && virtual_wll.second.y == other.virtual_wll.first.y));
    }
};

// 경로 데이터 구조체 정의
struct SEGDATA
{
    cv::Point centerPoint; // 기준이 되는 Point
    std::vector<cv::Point> feturePoints;
    std::vector<cv::Point> trajectoryPoints; // 경로를 저장하는 vector

    // 생성자
    SEGDATA() = default;
    SEGDATA(const cv::Point &key, const std::vector<cv::Point> &feture, const std::vector<cv::Point> &traj)
        : centerPoint(key), feturePoints(feture), trajectoryPoints(traj) {}
};


// Custom comparator for cv::Point
struct PointComparator
{
    bool operator()(const cv::Point &a, const cv::Point &b) const
    {
        return (a.x < b.x) || (a.x == b.x && a.y < b.y);
    }
};

// Custom comparator for LINEINFO to use in a set
struct LineInfoComparator
{
    bool operator()(const LINEINFO &a, const LINEINFO &b) const
    {
        return (PointComparator{}(a.virtual_wll.first, b.virtual_wll.first)) ||
              (a.virtual_wll.first == b.virtual_wll.first && PointComparator{}(a.virtual_wll.second, b.virtual_wll.second));
    }
};


class ROOMSEG
{
private:
  /* data */

  cv::Mat img_wall_;
  int rows_;
  int cols_;

  cv::Mat img_wall_rotated_;

  int rows_rot_;
  int cols_rot_;
  
  double angle_;

  
  Mat img_grid_; 
  Mat img_grid_skeletion_;

  std::vector<cv::Point> featurePts_;
  
  Mat img_raw_rotated_;  
  Mat img_freeSpace_;
  std::vector<cv::Point> trajectoryPts_;

  std::vector<std::vector<LINEINFO>> lineInfo_;

  int radius_ = 20;

  std::vector<LINEINFO> virtual_line_;  
  std::vector<cv::Rect> regions_box_;

  Mat img_fill_region_; 
  Mat img_label_;
  Mat img_segroom_;

  cv::Vec4i findLongestLine(const std::vector<cv::Vec4i> &lines);
  void zhangSuenThinning(const cv::Mat &src, cv::Mat &dst);
  void findConnectedComponents(const vector<Point> &contour, vector<vector<Point>> &components);

  Point calculateSnappedPoint(const Point& pixel, int gridSize);
  void gridSnapping(const Mat& inputImage, Mat& outputImage, int gridSize);

  bool isHalfOverlap(const Point &center1, int radius1, const Point &center2, int radius2); 
  bool isOverlap(const Point& center1, int radius1, const Point& center2, int radius2);

  vector<Point> addHalfOverlappingCircles(const vector<Point> &data, int radius);
  vector<Point> addNOverlappingCircles(const vector<Point>& data, int radius);

  double pointToLineDistance(const cv::Point &p, const cv::Point &lineStart, const cv::Point &lineEnd);
  bool isPointNearLine(const cv::Point &p, const cv::Point &lineStart, 
                        const cv::Point &lineEnd, double threshold);

  std::vector<LINEINFO> checkPointsNearLineSegments(const std::vector<cv::Point> &dataA, 
                                                  const std::vector<cv::Point> &dataB, 
                                                  double distance_threshold = 5.0);

  SEGDATA exploreFeaturePoint(std::vector<cv::Point> &feature_points,
                            std::vector<cv::Point> &trajectory_points,
                            const cv::Point &center, int radius);

  std::vector<cv::Point> findPointsInRange(const std::vector<cv::Point> &points,
                                            int x_min, int x_max,
                                            int y_min, int y_max);

  bool linesOverlap(const LINEINFO &line1, const LINEINFO &line2);
  std::vector<LINEINFO> fillterLine(std::vector<LINEINFO> &lines);

  std::vector<LINEINFO> convertToLineInfo(const std::vector<std::vector<LINEINFO>> &a);
  std::vector<LINEINFO> removeDuplicatesLines(const std::vector<LINEINFO> &lines);
  bool areEqualIgnoringOrder(const LINEINFO &a, const LINEINFO &b);

  void buildDataBase();
  void buildDataBaseTest(const cv::Mat& img_color_map);

  cv::Point findNearestBlackPoint(const cv::Mat& image, cv::Point center);
  //void regionGrowing(const cv::Mat &binaryImage, cv::Mat &output, cv::Point seed, uchar fillColor);

public:

  ROOMSEG(/* args */);
  ~ROOMSEG();

  cv::Mat getImgWall()
  {
    return img_wall_;
  }

  cv::Mat getImgWallRotated()
  {
    return img_wall_rotated_;
  }

  double getAngle()
  {
    return angle_;
  }

  cv::Mat getImgGridSnapping()
  {
    return img_grid_;
  }

  cv::Mat getImgGridSnapping2()
  {
    return img_grid_skeletion_;
  }

  std::vector<cv::Point> getFeaturePts()  
  {
    return featurePts_; 
  }

  cv::Mat getRotatedImage()
  {
    return img_raw_rotated_;
  }

  std::vector<cv::Point> getTrajectoryPts()
  {
    return trajectoryPts_; 
  }
  
  std::vector<LINEINFO> getVirtualLines()
  {
    return virtual_line_;
  }

  cv::Mat getImageFreeSpace()
  {
    return img_freeSpace_;
  }


  std::vector<cv::Rect> getRegionBox()
  {
    return regions_box_;
  }

  cv::Mat getLabelImage()
  {
    return img_label_;
  }

  cv::Mat getSegImage()
  {
    return img_segroom_;
  }



  void extractWallElements(const cv::Mat &occupancyMap, uchar thread_wall_value = 64);
  
  void makeRotatedImage();
  void makeRotatedImage(const cv::Mat& img_raw);
  void makeRotatedAngle();


  void makeGridSnappingContours(int length_contours=15, int gridSize=3);
  
  void extractFeaturePts();
  
  
  void extracTrajectorPts();

  void extractVirtualLine(double length_virtual_line = 22);
  void classificationRegon();

  //void segmentationRegion();
  void makeRegionContour();
  void segmentationRoom();

};



#endif //_ROOMSEG_H
