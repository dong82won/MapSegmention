#include <opencv2/opencv.hpp>

#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <stack>

//#include "lsd.h"
#include "featuredetection.h"
#include "trajectioryPoint.h"

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

cv::Mat makeFreeSpace(cv::Mat &src)
{
    int rows = src.rows;
    int cols = src.cols;

    cv::Mat dst = cv::Mat::zeros(rows, cols, CV_8UC1);

    for (int i=0; i<rows; i++)
    {
        for (int j=0; j<cols; j++)
        {
            uchar pixelValue = src.at<uchar>(i, j);
            if (pixelValue > 128) {
            //if (pixelValue > 205) {
                dst.at<uchar>(i, j) = 255;
            }
        }
    }
    return dst;
}


// 커스텀 비교 함수
struct PointCompare {
    bool operator()(const Point& lhs, const Point& rhs) const {
        if (lhs.x == rhs.x) {
            return lhs.y < rhs.y;
        }
        return lhs.x < rhs.x;
    }
};

// 거리 계산 함수
double calculateDistance(const Point &p1, const Point &p2)
{
    return sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2));
}

double euclideanDistance(const cv::Point &p1, const cv::Point &p2)
{
    return cv::norm(p1 - p2);
}
 

void exploreCircleLine(cv::Mat& image, const std::vector<cv::Point>& points, int radius) 
{
    if (points.empty()) return;

    // 원의 지름 계산
    int diameter = 1.5* radius;

    // 첫 번째 점에서 원을 그림
    cv::Point prevPoint = points[0];
    //cv::circle(image, prevPoint, radius, CV_RGB(0, 255, 0), 1);

    for (size_t i = 1; i < points.size(); i++) 
    {
        cv::Point currentPoint = points[i];

        // 이전 점과 현재 점 사이의 거리 계산
        double distance = cv::norm(currentPoint - prevPoint);

        // 거리가 반지름 두 배보다 크거나 같을 때만 원을 그림
        if (distance >= diameter) 
        {
            // 원을 그리는 위치 계산
            cv::circle(image, currentPoint, radius, CV_RGB(0, 255, 0), 1);
            
            // 반경 내의 모든 점을 탐색
            for (int y = -radius; y <= radius; ++y)
            {
                for (int x = -radius; x <= radius; ++x)
                {
                    // 현재 점이 원의 내부에 있는지 확인
                    if (x * x + y * y <= radius * radius) {
                        // 원의 내부에 있는 점에서 수행할 작업
                        cv::Point pointInCircle = currentPoint + cv::Point(x, y);
                        if (pointInCircle.x >= 0 && pointInCircle.x < image.cols &&
                            pointInCircle.y >= 0 && pointInCircle.y < image.rows)
                        {
                            // 이미지 범위를 벗어나지 않는 점에 대해서만 처리
                            image.at<cv::Vec3b>(pointInCircle) = cv::Vec3b(0, 0, 255); // 빨간색으로 점 찍기
                        }
                    }
                }
            }
            prevPoint = currentPoint;  // 이전 점을 업데이트
        }
    }
}


void exploreBoxLine(cv::Mat& image, const std::vector<cv::Point>& points, int radius, 
                    std::vector<cv::Rect>& boundingBoxes) 
{
    if (points.empty()) return;

    // 첫 번째 점에서 박스를 그림
    cv::Point prevPoint = points[0];
    cv::Rect initialBox(prevPoint.x - radius, prevPoint.y - radius, 2 * radius, 2 * radius);
    cv::rectangle(image, initialBox, CV_RGB(0, 255, 0), 1);    
    
    cv::Point center(initialBox.x + initialBox.width / 2, initialBox.y + initialBox.height / 2);
    circle(image, center, 3, Scalar(255, 255, 0), -1); // 빨간색 점

    boundingBoxes.push_back(initialBox); // 박스 저장

    for (size_t i = 1; i < points.size(); i+=radius) 
    {
        cv::Point currentPoint = points[i];

        // 이전 점과 현재 점 사이의 거리 계산
        double distance = cv::norm(currentPoint - prevPoint);

        // 거리가 반지름 두 배보다 크거나 같을 때만 박스를 그림
        if (distance >= 1.5 * radius) 
        {
            // 박스의 좌측 상단 점과 우측 하단 점 계산
            cv::Rect box(currentPoint.x - radius, currentPoint.y - radius, 2 * radius, 2 * radius);
            cv::rectangle(image, box, CV_RGB(0, 255, 0), 1);
            // 박스의 중심점 계산
            cv::Point center(box.x + box.width / 2, box.y + box.height / 2);
            // 중심점 표시
            circle(image, center, 3,  Scalar(255, 255, 0), -1); // 빨간색 점

            boundingBoxes.push_back(box); // 박스 저장            
            
            prevPoint = currentPoint;  // 이전 점을 업데이트
        }
    }
}

// // 점이 박스 안에 있는지 검사
// bool isPointInBoundingBox(const cv::Point& point, const std::vector<cv::Rect>& boundingBoxes)
// {
//     for (const auto& box : boundingBoxes)
//     {
//         if (box.contains(point))
//         {
//             return true;
//         }
//     }
//     return false;
// }
// 점이 박스 안에 있는지 검사
bool isPointInBoundingBox(const cv::Point& point, const cv::Rect& box)
{
    return box.contains(point);
}


// 점들을 순차적으로 정렬하여 실선 재구성
std::vector<cv::Point> sortPoints(const std::vector<cv::Point>& points) {
    std::vector<cv::Point> sortedPoints;
    if (points.empty()) return sortedPoints;

    std::vector<cv::Point> remainingPoints = points;
    cv::Point currentPoint = remainingPoints[0];
    sortedPoints.push_back(currentPoint);
    remainingPoints.erase(remainingPoints.begin());

    while (!remainingPoints.empty()) {
        auto nearestIt = std::min_element(remainingPoints.begin(), remainingPoints.end(),
            [&currentPoint](const cv::Point& p1, const cv::Point& p2) {
                return calculateDistance(currentPoint, p1) < calculateDistance(currentPoint, p2);
            });
        currentPoint = *nearestIt;
        sortedPoints.push_back(currentPoint);
        remainingPoints.erase(nearestIt);
    }

    return sortedPoints;
}
// 사용자 정의 비교 함수
struct RectCompare {
    bool operator()(const cv::Rect& lhs, const cv::Rect& rhs) const {
        if (lhs.y != rhs.y) return lhs.y < rhs.y;
        if (lhs.x != rhs.x) return lhs.x < rhs.x;
        if (lhs.height != rhs.height) return lhs.height < rhs.height;
        return lhs.width < rhs.width;
    }
};

int main()
{
    std::string home_path = getenv("HOME");
    //std::cout << home_path << std::endl;
    
    // 이미지 파일 경로
    cv::Mat raw_img = cv::imread(home_path + "/myStudyCode/MapSegmention/imgdb/occupancy_grid.png", cv::IMREAD_GRAYSCALE);
    //cv::Mat raw_img = cv::imread(home_path + "/myStudyCode/MapSegmention/imgdb/caffe_map.pgm", cv::IMREAD_GRAYSCALE);
    if (raw_img.empty())
    {
        std::cerr << "Error: Unable to open image file: " << std::endl;
        return -1;
    }
    cv::Mat result_img;
    cv::cvtColor(raw_img, result_img, cv::COLOR_GRAY2RGB);
    cv::Mat result_img2 = result_img.clone();

    // std::vector<cv::Point> featurePoints;
    // FeatureDetection fd(raw_img, featurePoints);
    
    FeatureDetection fd;
    //fd.straightLineDetection();

    //cv::imshow("img_lsd", img_lsd);


//     //--------------------------------------------------------------------------    
//     fd.detectEndPoints(img_lsd, 9);

//     std::vector<cv::Point> updata_featurePoints = fd.updateFeaturePoints();
//     //fd.imgShow(updata_featurePoints);

//     for (const auto &pt : updata_featurePoints)
//     {
//         cv::circle(result_img, pt, 3, cv::Scalar(0, 0, 255), -1 );
//     }


//     //--------------------------------------------------------------------------    
//     cv::Mat img_freeSpace = makeFreeSpace(raw_img);
//     cv::imshow("img_freeSpace", img_freeSpace);


//       // 이진화된 이미지를 반전
//     Mat invertedImg;
//     bitwise_not(img_freeSpace, invertedImg);
//     cv::imshow("invertedImg", invertedImg);

//     // 윤곽선 찾기
//     vector<vector<Point>> contours;
//     vector<Vec4i> hierarchy;
//     findContours(img_freeSpace, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);

//     cv::Mat color_img_freeSpace;
//     cv::cvtColor(img_freeSpace, color_img_freeSpace, cv::COLOR_GRAY2RGB);
    
//     for (size_t i = 0; i < contours.size(); ++i) {
//         std::cout <<"num: " <<contours[i].size() <<std::endl;
//         if (contours[i].size() > 10) {
//             drawContours(color_img_freeSpace, contours, static_cast<int>(i), Scalar(0, 255, 0), 2); 
//         }
//     }
//     cv::imshow("color_img_freeSpace", color_img_freeSpace);






//     TrajectionPoint tp;
//     cv::Mat img_dist= tp.makeDistanceTransform(img_freeSpace);

//     cv::Mat img_skeletion;
//     tp.zhangSuenThinning(img_dist, img_skeletion); 
//     cv::imshow("img_skeletion", img_skeletion);      

//     // 꺾이는 지점과 끝점을 저장할 벡터
//     std::vector<cv::Point> trajector_line;
//     std::vector<cv::Point> trajector_points = tp.extractBendingAndEndPoints(img_skeletion, trajector_line);         
//     std::vector<cv::Point> sort_trajector_line = sortPoints(trajector_line);


//     for (const auto &pt : sort_trajector_line)
//     {
//         cv::circle(result_img, pt, 1, cv::Scalar(255, 0, 255), -1 );
//     }



//     //cv::Mat test(img_skeletion.size(), CV_8UC3, CV_RGB(0, 0, 0));    
//     //for (const auto &pt : sort_trajector_line)
//     // for (size_t i =0; i< sort_trajector_line.size(); i+=10)
//     // {
//     //     cv::Point pt = sort_trajector_line[i];
//     //     cv::circle(test, pt, 1, cv::Scalar(255, 255, 255), -1 );        
    
//     // }
//     std::vector<cv::Rect> boundingBoxes;
//     exploreBoxLine(result_img, sort_trajector_line, 20, boundingBoxes);

//     // 박스에 점들을 매핑할 std::map
//     std::map<cv::Rect, std::vector<cv::Point>, RectCompare> boxToPointsMap;    
//  // 점들이 각 박스 안에 있는지 검사하고, std::map에 저장
//     for (const auto& box : boundingBoxes)
//     {
//         std::vector<cv::Point> pointsInBox;
//         for (const auto& pt : updata_featurePoints)
//         {
//             if (isPointInBoundingBox(pt, box))
//             {
//                 pointsInBox.push_back(pt);
//             }
//         }

//         if (!pointsInBox.empty())
//         {
//             if (pointsInBox.size() >= 2)
//                 boxToPointsMap[box] = pointsInBox;
//         }
//     }

//         // 결과 출력
//     for (const auto& entry : boxToPointsMap)
//     {
//         const cv::Rect& box = entry.first;
//         const std::vector<cv::Point>& points = entry.second;
//         cv::rectangle(result_img2, box, CV_RGB(0, 255, 0), 1);

//         std::cout << "박스 " << box << " 안에 있는 점들: ";
//         for (const auto& pt : points)
//         {
//             std::cout << pt << " ";
//             // 중심점 표시
//             circle(result_img2, pt, 3,  CV_RGB(255, 0, 0), -1); 
//         }
//         std::cout << std::endl;
//     }


//     cv::imshow("result_img2", result_img2);  
//     cv::imshow("result_img", result_img);    
    
    cv::waitKey();   
    return 0;
}

 
