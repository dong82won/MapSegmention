#include <opencv2/opencv.hpp>

#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <stack>
#include <algorithm>

// #include "lsd.h"
#include "featuredetection.h"
#include "trajectioryPoint.h"
#include "MSED.h"


using namespace cv;
using namespace std;


 

// 경로 데이터 구조체 정의
struct SEGDATA {
    
    cv::Point centerPoint;                // 기준이 되는 Point
    std::vector<cv::Point> feturePoints;
    std::vector<cv::Point> trajectoryPoints;   // 경로를 저장하는 vector
    bool state_ = false;

    
    std::vector<cv::Point> fetureCluster0;
    cv::Point2f cluster0Center;

    std::vector<cv::Point> fetureCluster1;
    cv::Point2f cluster1Center;
 

    // 생성자
    SEGDATA() = default;    
    SEGDATA(const cv::Point& key, const std::vector<cv::Point>& feture, const std::vector<cv::Point>& traj)
        : centerPoint(key), feturePoints(feture), trajectoryPoints(traj) {}

    void addState(bool state)
    {
        state_ = state;
    }

    // 경로에 포인트 추가
    void addfetureCluster0Pts(const cv::Point& point) {
        fetureCluster0.push_back(point);
    }

    void addfetureCluster1Pts(const cv::Point& point) {
        fetureCluster1.push_back(point);
    } 

    void addCluster0Center(const cv::Point2f& point){
        cluster0Center = point; 
    }
     
    void addCluster1Center(const cv::Point2f& point){
        cluster1Center = point; 
    } 

};
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

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
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

// 거리 계산 함수
double calculateDistance(const Point &p1, const Point &p2)
{
    return sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2));
}

double euclideanDistance(const cv::Point &p1, const cv::Point &p2)
{
    return cv::norm(p1 - p2);
}

// 거리 내의 점들을 병합하는 함수
void mergeClosePoints(std::vector<cv::Point> &points, int distanceThreshold)
{
    std::vector<cv::Point> mergedPoints;

    while (!points.empty())
    {
        cv::Point basePoint = points.back();
        points.pop_back();

        std::vector<cv::Point> closePoints;
        closePoints.push_back(basePoint);

        for (auto it = points.begin(); it != points.end();)
        {
            if (cv::norm(basePoint - *it) <= distanceThreshold)
            {
                closePoints.push_back(*it);
                it = points.erase(it);
            }
            else
            {
                ++it;
            }
        }

        // 평균 위치를 계산하여 병합된 점을 추가
        cv::Point avgPoint(0, 0);
        for (const auto &pt : closePoints)
        {
            avgPoint += pt;
        }
        avgPoint.x /= closePoints.size();
        avgPoint.y /= closePoints.size();
        mergedPoints.push_back(avgPoint);
    }

    points = mergedPoints;
}

// End-points를 감지하고 그 좌표를 vector에 저장
void detectEndPoints(const cv::Mat &skeleton, std::vector<cv::Point> &endPoints)
{
    for (int y = 1; y < skeleton.rows - 1; ++y)
    {
        for (int x = 1; x < skeleton.cols - 1; ++x)
        {
            if (skeleton.at<uchar>(y, x) == 255)
            {
                int count = 0;

                // 8방향 이웃 확인
                count += skeleton.at<uchar>(y - 1, x - 1) == 255 ? 1 : 0;
                count += skeleton.at<uchar>(y - 1, x) == 255 ? 1 : 0;
                count += skeleton.at<uchar>(y - 1, x + 1) == 255 ? 1 : 0;
                count += skeleton.at<uchar>(y, x + 1) == 255 ? 1 : 0;
                count += skeleton.at<uchar>(y + 1, x + 1) == 255 ? 1 : 0;
                count += skeleton.at<uchar>(y + 1, x) == 255 ? 1 : 0;
                count += skeleton.at<uchar>(y + 1, x - 1) == 255 ? 1 : 0;
                count += skeleton.at<uchar>(y, x - 1) == 255 ? 1 : 0;

                // End-point는 이웃이 하나만 있는 픽셀
                if (count == 1)
                {
                    endPoints.push_back(cv::Point(x, y));
                }
            }
        }
    }

    // 교차점 병합
    mergeClosePoints(endPoints, 12); // 3 픽셀 이내의 점을 병합
}


// 두 원형 영역이 겹치는지 확인하는 함수
bool isOverlap(const Point& center1, int radius1, const Point& center2, int radius2) {
    double distance = sqrt(pow(center1.x - center2.x, 2) + pow(center1.y - center2.y, 2));
    return distance < (radius1 + radius2);
}

// 두 원형 영역이 반만 겹치는지 확인하는 함수
bool isHalfOverlap(const Point& center1, int radius1, const Point& center2, int radius2) {
    double distance = sqrt(pow(center1.x - center2.x, 2) + pow(center1.y - center2.y, 2));
    return distance <= (radius1 + radius2) / 2.0;
}


// 주어진 중심과 반지름을 기반으로 원의 경계 점을 찾는 함수
vector<Point> edgePointsInCircle(const Point& center, int radius) {
    vector<Point> points;
    
    // 원의 경계에서 점들을 추출
    for (double angle = 0; angle < 2 * CV_PI; angle += 0.1) { // 각도를 0.1씩 증가시켜 점을 추출
        int x = static_cast<int>(center.x + radius * cos(angle));
        int y = static_cast<int>(center.y + radius * sin(angle));
        points.push_back(Point(x, y));
    }
    
    return points;
}


// 원형 탐색 범위를 추가하는 함수 
vector<Point> addHalfOverlappingCircles(const vector<Point>& data, int radius) {
    vector<Point> circlesCenters;
    
    for (const auto& point : data) {
        bool overlap = false;
        
        // 새로 추가할 원형 범위가 기존의 범위와 반만 겹치는지 확인
        for (const auto& existingCenter : circlesCenters) {
            if (isHalfOverlap(existingCenter, radius, point, radius)) {
                overlap = true;
                break;
            }
        }
        
        if (!overlap) {
            circlesCenters.push_back(point);
        }
    }
    
    return circlesCenters;
}


//추가
vector<Point> addNOverlappingCircles(const vector<Point>& data, int radius) {
    vector<Point> circlesCenters;
    
    for (const auto& point : data) {
        bool overlap = false;
        
        // 새로 추가할 원형 범위가 기존의 범위와 반만 겹치는지 확인
        for (const auto& existingCenter : circlesCenters) {
            if (isOverlap(existingCenter, radius, point, radius)) {
                overlap = true;
                break;
            }
        }
        
        if (!overlap) {
            circlesCenters.push_back(point);
        }
    }
    
    return circlesCenters;
}


vector<Point> addNonOverlappingCircles(const vector<Point>& data, int radius) {
    vector<Point> circlesCenters;

    // 첫 번째 점은 무조건 포함
    if (!data.empty()) {
        circlesCenters.push_back(data.front());
    }
    
    // 중간의 점들 처리
    for (size_t i = 1; i < data.size() - 1; ++i) {
        const auto& point = data[i];
        bool overlap = false;

        // 새로 추가할 원형 범위가 기존의 범위와 반만 겹치는지 확인
        for (const auto& existingCenter : circlesCenters) {
            if (isHalfOverlap(existingCenter, radius, point, radius)) {
                overlap = true;
                break;
            }
        }

        if (!overlap) {
            circlesCenters.push_back(point);
        }
    }

    // 마지막 점도 반드시 포함
    if (data.size() > 1) {
        circlesCenters.push_back(data.back());
    }
    
    return circlesCenters;
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


// 원의 내부를 채우는 함수
void fillCircle(Mat& image, const Point& center, int radius, const Scalar& color) {
    for (int y = center.y - radius; y <= center.y + radius; ++y) {
        for (int x = center.x - radius; x <= center.x + radius; ++x) {
            // 원의 방정식 (x - center.x)^2 + (y - center.y)^2 <= radius^2
            if ((x - center.x) * (x - center.x) + (y - center.y) * (y - center.y) <= radius * radius) {
                if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
                    image.at<Vec3b>(y, x) = Vec3b(color[0], color[1], color[2]); // 원 내부를 채움
                }
            }
        }
    }
}

void featureInCircle(Mat& image, const Point& center, int radius, const Scalar& color) {
    for (int y = center.y - radius; y <= center.y + radius; ++y) {
        for (int x = center.x - radius; x <= center.x + radius; ++x) {
            // 원의 방정식 (x - center.x)^2 + (y - center.y)^2 <= radius^2
            if ((x - center.x) * (x - center.x) + (y - center.y) * (y - center.y) <= radius * radius) {
                if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
                    image.at<Vec3b>(y, x) = Vec3b(color[0], color[1], color[2]); // 원 내부를 채움
                }
            }
        }
    }
}

// void detectExploreFeature(Mat &image, vector<Point> &fdata, const Point &center, int radius, const Scalar &color)
// {
//     if (fdata.empty())
//     {
//     }
//     else
//     {
//         bool state = 0;
//         for (int y = center.y - radius; y <= center.y + radius; ++y)
//         {
//             for (int x = center.x - radius; x <= center.x + radius; ++x)
//             {
//                 // 원의 방정식 (x - center.x)^2 + (y - center.y)^2 <= radius^2
//                 if ((x - center.x) * (x - center.x) + (y - center.y) * (y - center.y) <= radius * radius)
//                 {
//                     if (x >= 0 && x < image.cols && y >= 0 && y < image.rows)
//                     {

//                         for (size_t i = 0; i < fdata.size();)
//                         {
//                             cv::Point ft = fdata[i];
//                             if (x == ft.x && y == ft.y)
//                             {
//                                 std::cout << ft << std::endl;
//                                 fdata.erase(fdata.begin() + i);
//                                 state = 1;
//                             }
//                             else
//                             {
//                                 ++i;
//                             }
//                         }
//                     }
//                 }
//             }
//         }

//         if (state == 1)
//         {
//             std::cout << "center: " << center << std::endl;
//         }
//     }
// }

struct PointCompare {
    bool operator()(const cv::Point& p1, const cv::Point& p2) const {
        if (p1.x != p2.x)
            return p1.x < p2.x;
        return p1.y < p2.y;
    }
};

typedef std::map<cv::Point, std::vector<cv::Point>, PointCompare> PointMap;


PointMap detectExploreFeature(Mat &image, vector<Point> &fdata, const Point &center, int radius)
{
    PointMap dst;
    if (fdata.empty())
    {
        return dst;
    }

    std::vector<cv::Point> featurePts;
    for (int y = center.y - radius; y <= center.y + radius; ++y)
    {
        for (int x = center.x - radius; x <= center.x + radius; ++x)
        {
            // 원의 방정식 (x - center.x)^2 + (y - center.y)^2 <= radius^2
            if ((x - center.x) * (x - center.x) + (y - center.y) * (y - center.y) <= radius * radius)
            {
                if (x >= 0 && x < image.cols && y >= 0 && y < image.rows)
                {
                    cv::Point currentPoint(x, y);
                    if (std::find(fdata.begin(), fdata.end(), currentPoint) != fdata.end())
                    {
                        featurePts.push_back(currentPoint); 
                    }                    
                }
            }
        }
    }
    if(!featurePts.empty()) {
        dst[center] = featurePts;
    }
    return dst;
}


PointMap detectExploreFeature2(Mat &image, vector<Point> &fdata, const Point &center, int radius)
{
    PointMap dst;
    if (fdata.empty())
    {
        return dst;
    }

    std::vector<cv::Point> featurePts;
    for (int y = center.y - radius; y <= center.y + radius; ++y)
    {
        for (int x = center.x - radius; x <= center.x + radius; ++x)
        {
            // 원의 방정식 (x - center.x)^2 + (y - center.y)^2 <= radius^2
            if ((x - center.x) * (x - center.x) + (y - center.y) * (y - center.y) <= radius * radius)
            {
                if (x >= 0 && x < image.cols && y >= 0 && y < image.rows)
                {
                    cv::Point currentPoint(x, y);
                    if (std::find(fdata.begin(), fdata.end(), currentPoint) != fdata.end())
                    {
                        featurePts.push_back(currentPoint); 
                    }                    
                }
            }
        }
    }
    if(!featurePts.empty()) {
        dst[center] = featurePts;
    }
    return dst;
}

SEGDATA detectExploreFeature3(Mat &image, std::vector<cv::Point> sorted_trajectory_points, vector<Point> &fdata, const Point &center, int radius)
{

    SEGDATA dst2;
    //PointMap dst;
    if (fdata.empty())
    {
        return dst2;
    }

    std::vector<cv::Point> featurePts;
    std::vector<cv::Point> trajectoryPts;

    for (int y = center.y - radius; y <= center.y + radius; ++y)
    {
        for (int x = center.x - radius; x <= center.x + radius; ++x)
        {
            // 원의 방정식 (x - center.x)^2 + (y - center.y)^2 <= radius^2
            if ((x - center.x) * (x - center.x) + (y - center.y) * (y - center.y) <= radius * radius)
            {
                if (x >= 0 && x < image.cols && y >= 0 && y < image.rows)
                {
                    cv::Point explorePoint(x, y);                    
                    if (std::find(fdata.begin(), fdata.end(), explorePoint) != fdata.end())
                    {
                        featurePts.push_back(explorePoint); 
                    }                    
                }
            }
        }
    }

    if(!featurePts.empty()) {
        //dst[center] = featurePts;
        //dst2(center, featurePts, featurePts);
        dst2 = SEGDATA(center, featurePts, sorted_trajectory_points); // 올바른 객체 초기화
    }

    return dst2;
}


// x, y 범위 내에 포함되는 포인트를 찾는 함수
std::vector<cv::Point> findPointsInRange(const std::vector<cv::Point>& points, int x_min, int x_max, int y_min, int y_max) {
    
    std::vector<cv::Point> filteredPoints;

    // std::copy_if를 사용하여 조건에 맞는 점들만 필터링
    std::copy_if(points.begin(), points.end(), std::back_inserter(filteredPoints), 
        [x_min, x_max, y_min, y_max](const cv::Point& pt) {
            return (pt.x >= x_min && pt.x <= x_max && pt.y >= y_min && pt.y <= y_max);
        });

    return filteredPoints;
}

SEGDATA testExploreFeature3(std::vector<cv::Point> &feature_points, 
                            std::vector<cv::Point> &trajectory_points, 
                            const cv::Point &center, int radius)
{
    int x_min = center.x - radius;
    int x_max = center.x + radius;
    int y_min = center.y - radius;
    int y_max = center.y + radius;

    std::vector<cv::Point> featurePts = findPointsInRange(feature_points, x_min, x_max, y_min, y_max);
    std::vector<cv::Point> trajectoryPts = findPointsInRange(trajectory_points, x_min, x_max, y_min, y_max);

    SEGDATA dst = SEGDATA(center, featurePts, trajectoryPts);

    return dst;
}



void drawingSetpRectangle(Mat& image, cv::Point circlesCenters, int radius)
{
    int max_x = circlesCenters.x + radius;
    int max_y = circlesCenters.y + radius;
    int min_x = circlesCenters.x - radius;
    int min_y = circlesCenters.y - radius;
    cv::Rect rect(cv::Point(min_x, min_y), cv::Point(max_x, max_y));
 
    
    cv::rectangle(image, rect, cv::Scalar(255, 0, 0), 1);  // 파란색 사각형
    
}


void drawingSetpCircule(Mat& image, cv::Point circlesCenters, int radius)
{
    // 원형 영역을 이미지에 그리기
    vector<Point> edgePoints = edgePointsInCircle(circlesCenters, radius);
    // // 원의 경계 점을 초록색으로 표시
    // for (const auto& point : edgePoints) {
    //     if (point.x >= 0 && point.x < image.cols && point.y >= 0 && point.y < image.rows) {
    //         image.at<Vec3b>(point.y, point.x) = Vec3b(0, 255, 0);
    //     }
    // }

    // 원을 실선으로 그리기
    for (size_t i = 0; i < edgePoints.size(); i++)
    {
        Point start = edgePoints[i];
        Point end = edgePoints[(i + 1) % edgePoints.size()]; // 마지막 점과 첫 점을 연결
        line(image, start, end, Scalar(255, 0, 0), 1);       // 파란색으로 실선 그리기
    }
}

void drawingOutLineCircule(Mat& image, vector<Point> circlesCenters, int radius)
{

    // 원형 영역을 이미지에 그리기
    for (const auto& center : circlesCenters) {
        vector<Point> edgePoints = edgePointsInCircle(center, radius);
        
        // // 원의 경계 점을 초록색으로 표시
        // for (const auto& point : edgePoints) {
        //     if (point.x >= 0 && point.x < image.cols && point.y >= 0 && point.y < image.rows) {
        //         image.at<Vec3b>(point.y, point.x) = Vec3b(0, 255, 0);                 
        //     }
        // }        

        // 원을 실선으로 그리기
        for (size_t i = 0; i < edgePoints.size(); i++) {
            Point start = edgePoints[i];
            Point end = edgePoints[(i + 1) % edgePoints.size()]; // 마지막 점과 첫 점을 연결
            line(image, start, end, Scalar(255, 0, 0), 1); // 파란색으로 실선 그리기
        }
    }  
}

// 구조체 정의
struct PointPair {
    cv::Point p1;
    cv::Point p2;
    double dist; // 두 점 사이의 거리

    
          // 생성자
    PointPair(const cv::Point& point1, const cv::Point& point2)
        : p1(point1), p2(point2), dist(euclideanDistance(point1, point2)) {}
};

// FeaturePts와 TrajectoryPts 사이의 직교 거리를 계산하는 함수
std::pair<cv::Point, double> findClosestPoint(const cv::Point& featurePt, const std::vector<cv::Point>& trajectoryPts) {
    cv::Point closestPt;
    double minDistance = std::numeric_limits<double>::max();  // 매우 큰 값으로 초기화

    for (const auto& trajectoryPt : trajectoryPts) {
        double dist = euclideanDistance(featurePt, trajectoryPt);
        if (dist < minDistance) {
            minDistance = dist;
            closestPt = trajectoryPt;
        }
    }
    return {closestPt, minDistance};  // 가장 가까운 TrajectoryPt와 그 거리 반환
}


// 유클리드 거리를 계산하는 함수
double calculateEuclideanDistance(const cv::Point& pt) {
    return std::sqrt(pt.x * pt.x + pt.y * pt.y);
}

// 좌표를 원점 기준으로 정렬하는 함수
void sortPointsByDistanceFromOrigin(std::vector<cv::Point>& points) {
    std::sort(points.begin(), points.end(), [](const cv::Point& a, const cv::Point& b) {
        return calculateEuclideanDistance(a) < calculateEuclideanDistance(b);
    });
}



// 순차적으로 좌표를 정렬하는 함수
std::vector<cv::Point> sortPointsSequentially(const std::vector<cv::Point>& points) {
    if (points.empty()) return {};

    std::vector<cv::Point> sortedPoints;
    std::vector<cv::Point> remainingPoints = points;

    // 첫 좌표를 시작점으로 설정
    sortedPoints.push_back(remainingPoints[0]);
    remainingPoints.erase(remainingPoints.begin());

    // 남은 좌표들 중 가장 가까운 점을 선택하는 방식으로 순차적으로 정렬
    while (!remainingPoints.empty()) {
        const cv::Point& lastPoint = sortedPoints.back();
        auto nearestPointIter = std::min_element(remainingPoints.begin(), remainingPoints.end(),
            [&lastPoint](const cv::Point& a, const cv::Point& b) {
                return calculateDistance(lastPoint, a) < calculateDistance(lastPoint, b);
            });

        // 가장 가까운 점을 추가하고, 남은 리스트에서 제거
        sortedPoints.push_back(*nearestPointIter);
        remainingPoints.erase(nearestPointIter);
    }

    return sortedPoints;
}

// 극각(Polar Angle)을 계산하는 함수
double calculatePolarAngle(const cv::Point& pt) {
    return std::atan2(pt.y, pt.x);  // y축과 x축을 기준으로 각도 계산
}

// 좌표를 시계방향으로 정렬하는 함수
void sortPointsClockwise(std::vector<cv::Point>& points) {
    // atan2 값이 작은 것부터 큰 순서대로 정렬 (시계방향)
    std::sort(points.begin(), points.end(), [](const cv::Point& a, const cv::Point& b) {
        double angleA = calculatePolarAngle(a);
        double angleB = calculatePolarAngle(b);
        return angleA > angleB;  // 시계방향: 각도가 클수록 먼저
    });
}



// 원점에서 가장 가까운 점을 찾는 함수
cv::Point findClosestPointToOrigin(const std::vector<cv::Point>& points) {
    return *std::min_element(points.begin(), points.end(), [](const cv::Point& a, const cv::Point& b) {
        return calculateEuclideanDistance(a) < calculateEuclideanDistance(b);
    });
}

// 극각(Polar Angle)을 계산하는 함수 (기준점을 기준으로)
double calculatePolarAngleFromPoint(const cv::Point& reference, const cv::Point& pt) 
{
    double angle = std::atan2(pt.y - reference.y, pt.x - reference.x);
    // 각도를 양수로 조정 (시계방향 정렬을 위해)
    if (angle < 0) {
        angle += 2 * CV_PI;
    }
    return angle;
}

// 기준점에서 시계방향으로 좌표를 정렬하는 함수
void sortPointsClockwiseFromReference(std::vector<cv::Point>& points, const cv::Point& reference) {
    // 기준점에서 각도를 계산하여 시계방향으로 정렬
    std::sort(points.begin(), points.end(), [&reference](const cv::Point& a, const cv::Point& b) {
        double angleA = calculatePolarAngleFromPoint(reference, a);
        double angleB = calculatePolarAngleFromPoint(reference, b);
        return angleA > angleB;  // 시계방향: 큰 각도부터 정렬
    });
}


cv::Mat makeImageWall(cv::Mat &occupancyMap)
{

    int rows = occupancyMap.rows;
    int cols = occupancyMap.cols;

    cv::Mat wall_img = cv::Mat::zeros(occupancyMap.size(), CV_8UC1);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            uchar pixelValue = occupancyMap.at<uchar>(i, j);
            if (pixelValue < 64)
            {
                wall_img.at<uchar>(i, j) = 255;
            }
        }
    }
    return wall_img;
}


typedef struct _EDGEINFO
{ 
    cv::Point startPt;
    cv::Point endPt; 
    std::vector<cv::Point> turningPoints;
    std::vector<cv::Point> turningPoints2;

} EDGEINFO;





void drawPoints(cv::Mat &img, std::vector<Point> points, cv::Scalar color)
{
    for (int k = 0; k < points.size(); k++)
    {
        cv::drawMarker(img, points.at(k), color, cv::MARKER_TILTED_CROSS, 6);
    }
}
 

// 이웃을 카운트하는 함수
int countNeighbors(const cv::Point& point, const std::vector<cv::Point>& coordinates) {
    int count = 0;
    
    for (const auto& coord : coordinates) {
        // 8방향에 해당하는 이웃을 확인 (직교 및 대각선 방향)
        if (std::abs(coord.x - point.x) <= 1 && std::abs(coord.y - point.y) <= 1) {
            if (coord != point) {
                count++;
            }
        }
    }
    
    return count;
}

// 시작점과 끝점을 찾는 함수
std::pair<cv::Point, cv::Point> findStartEndPoints(const std::vector<cv::Point>& coordinates) {
    cv::Point startPoint(0, 0), endPoint(0, 0);
    
    for (const auto& point : coordinates) {
        int neighborCount = countNeighbors(point, coordinates);
        
        if (neighborCount == 1) {
            if (startPoint == cv::Point(0, 0)) {
                startPoint = point;  // 첫 번째로 찾은 점은 시작점
            } else {
                endPoint = point;    // 두 번째로 찾은 점은 끝점
            }
        }
    }
    
    return {startPoint, endPoint};
}


std::vector<cv::Point> findTurningPoints(const std::vector<cv::Point>& coords, int windowSize) {
    std::vector<cv::Point> turningPoints;

    for (size_t i = windowSize; i < coords.size() - windowSize; ++i) {
        // 기울기 계산을 위해 현재 윈도우의 좌표를 사용
        double avgSlope1 = 0.0;
        double avgSlope2 = 0.0;

        // 이전 윈도우의 평균 기울기
        for (int j = 1; j <= windowSize; ++j) {
            double slope = (coords[i].x - coords[i - j].x != 0) ? 
                           static_cast<double>(coords[i].y - coords[i - j].y) / (coords[i].x - coords[i - j].x) : 
                           std::numeric_limits<double>::infinity();
            avgSlope1 += slope;
        }
        avgSlope1 /= windowSize;

        // 이후 윈도우의 평균 기울기
        for (int j = 1; j <= windowSize; ++j) {
            double slope = (coords[i + j].x - coords[i].x != 0) ? 
                           static_cast<double>(coords[i + j].y - coords[i].y) / (coords[i + j].x - coords[i].x) : 
                           std::numeric_limits<double>::infinity();
            avgSlope2 += slope;
        }
        avgSlope2 /= windowSize;

        double diff_slop = std::abs(avgSlope1 - avgSlope2);
        cout <<"diff_slop: " << diff_slop << ", avgSlope1: "  << avgSlope1 << " ,avgSlope2: "  << avgSlope2 <<  endl;

        // 평균 기울기 변화가 있을 경우 꺾이는 지점 추가
        //if (avgSlope1 == std::numeric_limits<double>::infinity() && avgSlope2 == std::numeric_limits<double>::infinity())            
        if (diff_slop != std::numeric_limits<double>::infinity())        
        {
            if (diff_slop > 0.5)
                turningPoints.emplace_back(coords[i].x, coords[i].y);
        }
    }

    mergeClosePoints(turningPoints, 6);
    return turningPoints;
}


std::vector<cv::Point> findTurningChainPoints(const std::vector<cv::Point>& coords, int windowSize)
{
    std::vector<cv::Point> turningPoints;

    for (size_t i = windowSize; i < coords.size() - windowSize; ++i) {
        double avgDir1 = 0.0;
        double avgDir2 = 0.0;

        // 이전 윈도우의 평균 방향
        for (int j = 1; j <= windowSize; ++j) {
            double dir = std::atan2(coords[i].y - coords[i - j].y, coords[i].x - coords[i - j].x);
            avgDir1 += dir;
        }
        avgDir1 /= windowSize;

        // 이후 윈도우의 평균 방향
        for (int j = 1; j <= windowSize; ++j) {
            double dir = std::atan2(coords[i + j].y - coords[i].y, coords[i + j].x - coords[i].x);
            avgDir2 += dir;
        }
        avgDir2 /= windowSize;

        // 방향 차이를 계산하고 각도 변환
        double angleDiff = std::abs(avgDir1 - avgDir2) * 180.0 / M_PI;

        cout << "angleDiff: "<< angleDiff << endl;
        // 각도 차이가 45도 이상일 때 꺾이는 지점 추가
        if (angleDiff >= 60.0) {
            turningPoints.emplace_back(coords[i].x, coords[i].y);
        }
    }

    
    mergeClosePoints(turningPoints, 6);
    return turningPoints;
}

// 중복되는 cv::Point 데이터를 제거하는 함수
void removeDuplicatePoints(std::vector<cv::Point>& points) {
    // points를 정렬
    std::sort(points.begin(), points.end(), [](const cv::Point& a, const cv::Point& b) {
        return (a.x < b.x) || (a.x == b.x && a.y < b.y);
    });

    // 중복된 요소를 points의 끝으로 이동
    auto last = std::unique(points.begin(), points.end());

    // 중복된 요소를 제거
    points.erase(last, points.end());
} 
 

// // B 데이터에 가장 가까운 A 데이터의 점을 찾는 함수
// cv::Point findClosestPoint(const std::vector<cv::Point>& A, const std::vector<cv::Point>& B) {
//     double minDistance = std::numeric_limits<double>::max();
//     cv::Point closestPoint;

//     for (const auto& b : B) {
//         for (const auto& a : A) {
//             double distance = calculateDistance(b, a);
//             if (distance < minDistance) {
//                 minDistance = distance;
//                 closestPoint = a;
//             }
//         }
//     }

//     return closestPoint;
// }

bool isPointAboveLine(const cv::Point& point, cv::Point& lineStart, cv::Point& lineEnd) {
    double lineEquation = (lineEnd.y - lineStart.y) * (point.x - lineStart.x) - 
                          (lineEnd.x - lineStart.x) * (point.y - lineStart.y);
    return lineEquation < 0; // 음수이면 아래쪽
}

void splitPointsByLine(std::vector<cv::Point>& A, std::vector<cv::Point>& B, 
                       std::vector<cv::Point>& aboveLine, std::vector<cv::Point>& belowLine) {
    for (const auto& point : A) {
        bool isAbove = false;

        for (size_t i = 0; i < B.size() - 1; ++i) {
            if (isPointAboveLine(point, B[i], B[i + 1])) {
                isAbove = true;
                break;
            }
        }

        if (isAbove) {
            belowLine.push_back(point); // 아래쪽으로 수정
        } else {
            aboveLine.push_back(point); // 위쪽으로 수정
        }
    }
}


void bifurcatePoints(const std::vector<cv::Point>& A, const std::vector<cv::Point>& B, 
                     std::vector<cv::Point>& A_above, std::vector<cv::Point>& A_below,
                     std::vector<cv::Point>& A_left, std::vector<cv::Point>& A_right) {
    // A의 모든 점을 확인
    for (const auto& a_point : A) {
        bool above = true, below = true, left = true, right = true;

        // B의 점들을 사용하여 A 점의 위치를 확인
        for (size_t i = 0; i < B.size() - 1; ++i) {
            cv::Point b1 = B[i];
            cv::Point b2 = B[i + 1];

            // 선의 방정식으로 A 점의 위치 확인
            float slope = (float)(b2.y - b1.y) / (b2.x - b1.x);
            float intercept = b1.y - slope * b1.x;
            float line_y = slope * a_point.x + intercept;

            if (a_point.y > line_y) {
                below = false;  // A 점이 B 선 위쪽에 있음
            } else {
                above = false;  // A 점이 B 선 아래쪽에 있음
            }

            // 좌우 판단
            if (a_point.x > b1.x && a_point.x > b2.x) {
                left = false;  // A 점이 B 선의 왼쪽에 있음
            } else if (a_point.x < b1.x && a_point.x < b2.x) {
                right = false;  // A 점이 B 선의 오른쪽에 있음
            }
        }

        // A 점을 그룹에 추가
        if (above) {
            A_above.push_back(a_point);
        } else if (below) {
            A_below.push_back(a_point);
        }
        if (left) {
            A_left.push_back(a_point);
        } else if (right) {
            A_right.push_back(a_point);
        }
    }
}


// A 데이터의 모든 점에 대해 B 데이터의 각 점과 가장 가까운 거리를 계산하는 함수
double findMinDistanceToContour(const cv::Point& ptB, const std::vector<cv::Point>& contourA) {
    double minDistance = calculateDistance(ptB, contourA[0]);
    for (const auto& ptA : contourA) {
        double distance = calculateDistance(ptB, ptA);
        if (distance < minDistance) {
            minDistance = distance;
        }
    }
    return minDistance;
}



// 두 점 간의 각도 계산 함수 (라디안 단위)
double calculateAngle(const cv::Point& p1, const cv::Point& p2) {
    return atan2(p2.y - p1.y, p2.x - p1.x) * 180 / CV_PI; // 라디안을 도로 변환
}

// 내각 계산 함수
double calculateInternalAngle(const cv::Point& center, const cv::Point& point1, const cv::Point& point2) {
    double angle1 = calculateAngle(center, point1);
    double angle2 = calculateAngle(center, point2);
    
    double internalAngle = angle2 - angle1;
    if (internalAngle < 0) {
        internalAngle += 360; // 0에서 360도 범위로 조정
    }

    return internalAngle;
}

// // 모든 점 간의 거리와 내각 계산 함수
// void calculateDistancesAndAngles(const std::vector<cv::Point>& points, const cv::Point& center) {
//     for (size_t i = 0; i < points.size(); ++i) {
//         for (size_t j = i + 1; j < points.size(); ++j) {
//             const cv::Point& point1 = points[i];
//             const cv::Point& point2 = points[j];

//             // 두 점 간의 거리
//             double distance = calculateDistance(point1, point2);

//             // 내각 계산
//             double internalAngle = calculateInternalAngle(center, point1, point2);

//             std::cout << "Point " << i + 1 << " to Point " << j + 1 << " distance: "
//                       << distance << ", Internal Angle: " << internalAngle << "°" << std::endl;
//         }
//     }
// }
 


// 두 선분이 교차하는지 확인하는 함수
bool linesIntersect(const cv::Point& A1, const cv::Point& A2, const cv::Point& B1, const cv::Point& B2) {
    auto cross = [](const cv::Point& p1, const cv::Point& p2) {
        return p1.x * p2.y - p1.y * p2.x;
    };

    cv::Point A = A2 - A1;
    cv::Point B = B2 - B1;
    cv::Point C = B1 - A1;

    double d1 = cross(A, C);
    double d2 = cross(A, B);
    double d3 = cross(B, C);
    double d4 = cross(B, A);

    if (d2 == 0 && d4 == 0) {
        // Collinear case
        return (std::min(A1.x, A2.x) <= std::max(B1.x, B2.x) && 
                std::max(A1.x, A2.x) >= std::min(B1.x, B2.x) && 
                std::min(A1.y, A2.y) <= std::max(B1.y, B2.y) && 
                std::max(A1.y, A2.y) >= std::min(B1.y, B2.y));
    }
    return (d1 * d2 < 0) && (d3 * d4 < 0);
}

// 모든 점 간의 거리 계산 함수
void calculateDistancesWithBIntersect(const std::vector<cv::Point>& pointsA, const std::vector<cv::Point>& pointsB) {
    std::vector<std::string> results; // 결과를 저장할 벡터

    for (size_t i = 0; i < pointsA.size(); ++i) {
        for (size_t j = i + 1; j < pointsA.size(); ++j) {
            const cv::Point& pointA1 = pointsA[i];
            const cv::Point& pointA2 = pointsA[j];

            bool passedThroughB = false;
            double distance = euclideanDistance(pointA1, pointA2);

            // 특정 점 쌍에 대한 예외 처리
            if ((pointA1.x == 50 && pointA1.y == 144 && pointA2.x == 64 && pointA2.y == 130) ||
                (pointA1.x == 64 && pointA1.y == 130 && pointA2.x == 50 && pointA2.y == 144)) {
                passedThroughB = true; // 강제로 교차 처리
            }

            for (size_t k = 0; k < pointsB.size() - 1; ++k) {
                if (linesIntersect(pointA1, pointA2, pointsB[k], pointsB[k + 1])) {
                    passedThroughB = true;
                    break; // 첫 번째 교차점을 찾았을 경우 루프 탈출
                }
            }

            // 결과를 벡터에 추가
            if (passedThroughB) {
                results.push_back("Point A (" + std::to_string(pointA1.x) + ", " + std::to_string(pointA1.y) + ") to Point A (" 
                                  + std::to_string(pointA2.x) + ", " + std::to_string(pointA2.y) + ") skipped.");
            } else {
                results.push_back("Point A (" + std::to_string(pointA1.x) + ", " + std::to_string(pointA1.y) + ") to Point A (" 
                                  + std::to_string(pointA2.x) + ", " + std::to_string(pointA2.y) + "), Distance: " 
                                  + std::to_string(distance));
            }
        }
    }

    // 결과 출력
    for (const auto& result : results) {
        std::cout << result << std::endl;
    }
}



int main() {
    std::string home_path = getenv("HOME");
    // std::cout << home_path << std::endl;

    // 이미지 파일 경로
    cv::Mat raw_img  = cv::imread(home_path + "/myWorkCode/MapSegmention/imgdb/occupancy_grid.png", cv::IMREAD_GRAYSCALE);
    //cv::Mat wall_img = cv::imread(home_path + "/myWorkCode/MapSegmention/imgdb/occupancy_grid_wall.png", cv::IMREAD_GRAYSCALE);
    //cv::imshow("wall_img", wall_img);
    //cv::Mat raw_img = cv::imread(home_path + "/myWorkCode/regonSeg/imgdb/caffe_map.pgm", cv::IMREAD_GRAYSCALE);
    if (raw_img.empty())
    {
        std::cerr << "Error: Unable to open image file: " << std::endl;
        return -1;
    }

    cv::Mat img_wall = makeImageWall(raw_img);//(occupancyMap);    
    imshow("occupancyMap", img_wall);


    // detect edge line
    std::vector<std::vector<cv::Point>> edgeChains;
    uchar *occupancy_map = img_wall.data;    
    int rows = img_wall.rows;
    int cols = img_wall.cols;
    MSED edgeDetector;
    edgeDetector.MSEdge(occupancy_map, rows, cols, edgeChains);

    cout << "edgeChains.size(): " << edgeChains.size() << endl;
    cv::Mat edgeImg = cv::Mat::zeros(rows, cols, CV_8UC1);
     
    std::vector<EDGEINFO> ptData;
    for (int i = 0; i < edgeChains.size(); i++)
    {
        //cout << "edgeChainsPixel.size(): " << edgeChains[i].size() << endl;

        for (int j = 0; j < edgeChains[i].size(); j++)
        {
            // edgeChains 좌표 변경해야 됨
            int y0 = edgeChains[i][j].y;
            int x0 = edgeChains[i][j].x;
            edgeImg.at<uchar>(y0, x0) = 255;
            //std::cout << "(" << x0 << ", " << y0 << ")";              
        }        

        
        // cv::imshow("edgeImg", edgeImg);
        // cv::waitKey(0); 

        
        int index_end = edgeChains[i].size()-1;
        cv::Point startPt = edgeChains[i][0];        
        cv::Point endPt = edgeChains[i][index_end];

        
        EDGEINFO pts;
        pts.startPt = startPt;
        pts.endPt = endPt;
        
        pts.turningPoints = findTurningPoints(edgeChains[i], 6);
        pts.turningPoints2 = findTurningChainPoints(edgeChains[i], 6);        
        ptData.push_back(pts);  
    }

    cv::Mat edgeImgColor;
    cv::cvtColor(edgeImg, edgeImgColor, COLOR_GRAY2BGR);

    std::vector<cv::Point> featurePts; 
    for (size_t i =0; i< ptData.size(); i++)
    {

        cv::Point s = ptData[i].startPt;
        cv::Point e = ptData[i].endPt;

        cv::circle(edgeImgColor, s, 3, CV_RGB(255, 0, 0), -1);
        cv::circle(edgeImgColor, e, 3, CV_RGB(255, 0, 0), -1);
        
        featurePts.push_back(Point(ptData[i].startPt));
        featurePts.push_back(Point(ptData[i].endPt));

        for (size_t j =0; j <ptData[i].turningPoints.size(); j++)
        {   
            featurePts.push_back(ptData[i].turningPoints[j]);
            cv::circle(edgeImgColor, ptData[i].turningPoints[j], 3, CV_RGB(0, 255, 0), -1);            
        }

        for (size_t j =0; j <ptData[i].turningPoints2.size(); j++)
        {
            featurePts.push_back(ptData[i].turningPoints2[j]);
            cv::circle(edgeImgColor, ptData[i].turningPoints2[j], 3, CV_RGB(0, 255, 255), -1);            
        } 
    }
    // 중복 제거
    removeDuplicatePoints(featurePts);


    cv::imshow("edgeImgColor", edgeImgColor);  
    imshow("edgeImg", edgeImg);

    //namedWindow("occupancyMap_color", WINDOW_KEEPRATIO && WINDOW_AUTOSIZE);
    //imshow("occupancyMap_color", occupancyMap_color);

  
    cv::Mat result_img;
    cv::cvtColor(raw_img, result_img, cv::COLOR_GRAY2RGB); 
    cv::Mat result_img2 = result_img.clone();
   
    // for (size_t i = 0; i< featurePts.size(); i++)
    // {
    //     cv::Point pt = featurePts[i];
    //     cv::circle(result_img, pt, 3, CV_RGB(0, 255, 0), -1); 
    
    // }
    // cv::imshow("result_img", result_img);

    
    cv::Mat img_freeSpace = makeFreeSpace(raw_img);
    imshow("img_freeSpace", img_freeSpace);
 
    //-----------------------------------------------------

    TrajectionPoint tp;
    cv::Mat img_dist = tp.makeDistanceTransform(img_freeSpace);
    imshow("img_dist", img_dist);

    cv::Mat img_skeletion;
    tp.zhangSuenThinning(img_dist, img_skeletion); 
    cv::imshow("img_skeletion", img_skeletion);      

    std::vector<cv::Point> trajectory_points;
    for (int i = 0; i<img_skeletion.rows; i++) {
        for (int j=0; j<img_skeletion.cols; j++) {

            if (img_skeletion.at<uchar>(i, j) == 255) {
                trajectory_points.push_back(cv::Point(j, i));
            }
        }
    }

    cv::Mat img_color;
    cv::cvtColor(img_dist, img_color, cv::COLOR_GRAY2RGB);

    // for (const auto &pt : trajector_points)
    // {
    //     cv::circle(img_color, pt, 3, cv::Scalar(0, 0, 255), -1);
    //     // cv::imshow("img_color", img_color);      
    //     // cv::waitKey();
    // }

    std::vector<cv::Point> sorted_trajectory_points = sortPoints(trajectory_points);
    //removeDuplicatePoints(sorted_trajectory_points);

    for (const auto &pt : sorted_trajectory_points)
    {
        cv::circle(result_img, pt, 1, CV_RGB(255, 0, 0), -1);
    }

    int radius = 20; // 탐색 범위 반지름
    vector<Point> circlesCenters = addHalfOverlappingCircles(sorted_trajectory_points, radius);
    //vector<Point> circlesCenters = addNOverlappingCircles(sorted_trajectory_points, radius);

    // # 1
    // for (size_t i = 0; i < circlesCenters.size(); i++)
    // {
    //     cv::Point cp = circlesCenters[i];
    //     PointMap db = detectExploreFeature(result_img, fpoints, cp, radius);

    //     for (const auto &pair : db)
    //     {
    //         const cv::Point &key = pair.first;
    //         const std::vector<cv::Point> &value = pair.second;

    //         if (value.size() > 1)
    //         {
    //             drawingSetpCircule(result_img, cp, radius);
    //             std::cout << "Key Point: (" << key.x << ", " << key.y << ")\n";
    //             std::cout << "Values: ";
    //             for (const auto &pt : value)
    //             {
    //                 std::cout << "(" << pt.x << ", " << pt.y << ") ";
    //             }
    //             std::cout << std::endl;
    //         }
    //     }

    std::vector<cv::Point> fpts;
    std::vector<cv::Point> tpts;
 
    std::vector<SEGDATA> database; 

    
    cv::namedWindow("result_img0", WINDOW_KEEPRATIO && WINDOW_AUTOSIZE);
    for (size_t i = 0; i < circlesCenters.size(); )
    {
        
        cv:Mat img_update = result_img.clone();

        cv::Point cp = circlesCenters[i];
 
        cv::drawMarker(img_update, cp, cv::Scalar(0, 0, 255), cv::MARKER_CROSS);
        SEGDATA db = testExploreFeature3(featurePts, sorted_trajectory_points, cp, radius);
                    
        //지정한 윈도우 안에 feture point 2개 이하 이며 탐색 제외
        if ( db.feturePoints.size() < 2)  {  
            circlesCenters.erase(circlesCenters.begin() + i);  
        } else {
 
            //drawingSetpCircule(result_img, cp, radius); 
            drawingSetpRectangle(img_update, cp, radius);

            db.addState(true);
            std::cout << "-----------------------------------------------------" <<std::endl;
            std::cout << "Center Point:(" << db.centerPoint.x << ", " << db.centerPoint.y << ")\n";            
             
            std::cout << "TrajectoryPts:";
            for (const auto& pt : db.trajectoryPoints) {

                cv::circle(img_update, pt, 1, cv::Scalar(255, 0, 0), -1);
                std::cout << "(" << pt.x << ", " << pt.y << ") ";
            };
            std::cout <<std::endl;
            std::cout << "FeaturePt:\n";
            for (const auto &pt : db.feturePoints)
            { 
                cv::circle(img_update, pt, 3, cv::Scalar(0, 255, 0), -1);
                std::cout << "(" << pt.x << ", " << pt.y << ") ";
            } 

            //---------------------------------------------------------------------------------------------------------
            
            std::cout <<std::endl;
            // 거리 계산 함수 호출
            calculateDistancesWithBIntersect(db.feturePoints, db.trajectoryPoints);







            std::cout <<std::endl;
            std::cout <<"-----------------------------------------------------" <<std::endl;   
            ++i; 

            database.push_back(db);
        } 
        cv::imshow("result_img0", img_update); 
        cv::waitKey(0); 
     
     
    }

    /*
    cv::Mat dst;
    double scaleFactor = 1.5;
    cv::resize(result_img, dst, cv::Size(), scaleFactor, scaleFactor);
    cv::imshow("dst", dst);
    */

    //  // 외곽선 검출
    // std::vector<std::vector<cv::Point>> contours;
    // std::vector<cv::Vec4i> hierarchy;

    // cv::findContours(img_freeSpace, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    
    // // 외곽선을 그리기 위한 컬러 이미지로 변환
    // cv::Mat contourImage = cv::Mat::zeros(img_freeSpace.size(), CV_8UC3);
    // cv::Mat contourImage2 = contourImage.clone();

    // for (size_t i = 0; i < contours.size(); i++) {

    //     std::vector<cv::Point> approx;
    //     double epsilon = 0.001 * cv::arcLength(contours[i], true); 
    //     cv::approxPolyDP(contours[i], approx, epsilon, true);

    //     double perimeter = cv::arcLength(approx, true);  // 근사화된 외곽선의 길이
    //     double area = cv::contourArea(approx);  // 근사화된 외곽선의 면적 

    //     //std::cout << "1 Contour " << i << " has length: " << perimeter << " and area: " << area << std::endl;
    //     cv::drawContours(contourImage, std::vector<std::vector<cv::Point>>{approx}, -1, cv::Scalar(0, 255, 0), 1);  

    //     if (area > 15 && perimeter > 15 ) 
    //     {
    //         cv::drawContours(contourImage2, std::vector<std::vector<cv::Point>>{approx}, -1, cv::Scalar(0, 255, 0), 1);  
    //         std::cout << "2 Contour " << i << " has length: " << perimeter << " and area: " << area << std::endl;
    //     }
        
    // }


    // cv::imshow("Contours", contourImage);
    // cv::imshow("contours2", contourImage2);
    
    cv::waitKey();







        return 0;
    }

