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


using namespace cv;
using namespace std;


// 경로 데이터 구조체 정의
struct SEGDATA {
    
    cv::Point centerPoint;                // 기준이 되는 Point
    std::vector<cv::Point> feturePoints;
    std::vector<cv::Point> trajectoryPoints;   // 경로를 저장하는 vector
    bool state_ = false;

    // 생성자
    SEGDATA() = default;    
    SEGDATA(const cv::Point& key, const std::vector<cv::Point>& feture, const std::vector<cv::Point>& traj)
        : centerPoint(key), feturePoints(feture), trajectoryPoints(traj) {}

    void addState(bool state)
    {
        state_ = state;
    }

    // // 경로에 포인트 추가
    // void addFeturePoints(const cv::Point& point) {
    //     feturePoints.push_back(point);
    // }

    // void addTrajectoryPoint(const cv::Point& point) {
    //     trajectoryPoints.push_back(point);
    // } 
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
double calculatePolarAngleFromPoint(const cv::Point& reference, const cv::Point& pt) {
    return std::atan2(pt.y - reference.y, pt.x - reference.x);  // 기준점에서의 각도 계산
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


    int rows = raw_img.rows;
    int cols = raw_img.cols;
 
    cv::Mat wall(rows, cols, CV_8UC1, cv::Scalar(255));

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            uchar pixelValue = raw_img.at<uchar>(i, j);
            if (pixelValue < 64) { 
                wall.at<uchar>(i, j) = 0;
            }
        }
    }
    
    
    cv::imshow("wall", wall);

    // 모폴로지 연산을 위한 커널 생성 (3x3 사각형 커널)
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    //cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

    // 모폴로지 열림 연산 (작은 노이즈 제거)
    cv::Mat opened;
    cv::morphologyEx(wall, opened, cv::MORPH_OPEN, kernel);
 
    
    cv::imshow("opened", opened);
    
    FeatureDetection fd(opened);
    fd.straightLineDetection();

    // 병합 픽셀 설정: 9, 12;
    fd.detectEndPoints(9);
    std::vector<cv::Point> fpoints = fd.getUpdateFeaturePoints();


    cv::Mat result_img;
    cv::cvtColor(raw_img, result_img, cv::COLOR_GRAY2RGB);

    cv::Mat result_img2 = result_img.clone();

    cv::Mat result_img3 = result_img.clone();



    cv::line(result_img2, cv::Point(119, 91), cv::Point(105, 87),  cv::Scalar(0, 0, 255), 2);
    cv::line(result_img2, cv::Point(105, 87), cv::Point(122, 83),  cv::Scalar(0, 0, 255), 2);
    cv::line(result_img2, cv::Point(122, 83), cv::Point(94, 72) ,  cv::Scalar(0, 0, 255), 2);
    
    cv::line(result_img3, cv::Point(105, 87), cv::Point(94, 72),  cv::Scalar(0, 255, 0), 2);
    cv::line(result_img3, cv::Point(94, 72), cv::Point(119, 91),  cv::Scalar(0, 255, 0), 2);
    cv::line(result_img3, cv::Point(119, 91), cv::Point(122, 83),  cv::Scalar(0, 255, 0), 2);
    
 

    cv::imshow("result_img2-1", result_img2);
    cv::imshow("result_img3", result_img3);


    for (size_t i = 0; i< fpoints.size(); i++)
    {
        cv::Point pt = fpoints[i];;
        cv::circle(result_img, pt, 3, CV_RGB(0, 255, 0), -1); 
    }
    
    cv::imshow("result_img", result_img);

    
    cv::Mat img_freeSpace = makeFreeSpace(raw_img);
    imshow("img_freeSpace", img_freeSpace);

    //-----------------------------------------------------
 

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
    
    for (const auto &pt : sorted_trajectory_points)
    {
        cv::circle(result_img, pt, 1, CV_RGB(255, 0, 0), -1);
    }

    int radius = 20; // 탐색 범위 반지름
    //vector<Point> circlesCenters = addHalfOverlappingCircles(sorted_trajectory_points, radius);
    vector<Point> circlesCenters = addNOverlappingCircles(sorted_trajectory_points, radius);

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
     
    for (size_t i = 0; i < circlesCenters.size(); )
    {
        cv::Point cp = circlesCenters[i];
        SEGDATA db = testExploreFeature3( fpoints, sorted_trajectory_points, cp, radius);
                    
        
        if ( db.feturePoints.size() < 2)  { 

            circlesCenters.erase(circlesCenters.begin() + i); 
        
        } else {

            
            //drawingSetpCircule(result_img, cp, radius); 
            drawingSetpRectangle(result_img, cp, radius);

            db.addState(true);
            std::cout << "-----------------------------------------------------" <<std::endl;
            std::cout << "Key Point:(" << db.centerPoint.x << ", " << db.centerPoint.y << ")\n";            
            
            
            /*
            // #2
            // for (const auto &pt : db.feturePoints)
            // {
            //     //std::cout << "(" << pt.x << ", " << pt.y << ") "; 

            //     std::pair<cv::Point, double> result = findClosestPoint(pt, db.trajectoryPoints);  // auto 대신 std::pair 사용
            //     cv::Point closestPt = result.first;
            //     double distance = result.second;

            //     // 결과 출력
            //     std::cout << "FeaturePt:(" << pt.x << ", " << pt.y << ") -> ";
            //     std::cout << "Closest TrajectoryPt: (" << closestPt.x << ", " << closestPt.y << ") ";
            //     std::cout << "Distance: " << distance << std::endl;

            //     cv::line(result_img, cv::Point(pt), cv::Point(closestPt), CV_RGB(255, 0, 0), 1);

            // };  
            */
            
            std::cout << "FeaturePt:\n"; 
            for (const auto &pt : db.feturePoints)
            {
                std::cout << "("<< pt.x << ", " << pt.y << ") "; 
            }
            std::cout << std::endl;


             // // 좌표를 순차적으로 정렬
            // std::vector<cv::Point> sortedPoints = sortPointsSequentially(db.feturePoints); 
            // for (const auto &pt : sortedPoints)
            // {
            //     std::cout << "("<< pt.x << ", " << pt.y << ") "; 
            // }


            // 좌표를 원점 기준으로 정렬
            //sortPointsByDistanceFromOrigin(db.feturePoints); 
            
            // 좌표를 시계방향으로 정렬
            //sortPointsClockwise(db.feturePoints); 


            // 원점에서 가장 가까운 점을 찾기
            cv::Point closestPoint = findClosestPointToOrigin(db.feturePoints);

            // 가장 가까운 점을 기준으로 시계방향 정렬
            sortPointsClockwiseFromReference(db.feturePoints, closestPoint);

            for (const auto &pt : db.feturePoints)
            {
                std::cout << "("<< pt.x << ", " << pt.y << ") "; 
            }
 

            std::cout << std::endl;
            int numPoints = db.feturePoints.size();  
            // 쌍을 정의 (PointPair 사용)
            std::vector<PointPair> pairs;  
            if (numPoints > 2) 
            {

                for (int j=0; j<numPoints; ++j)
                {
                    cv::Point p1 = db.feturePoints[j];
                    cv::Point p2 = db.feturePoints[(j + 1) % numPoints]; // 다음 점, 마지막 점에서 첫 점으로 순환
                    pairs.emplace_back(p1, p2);
                } 
            }
            else {
                for (int j=0; j<numPoints-1; ++j)
                {
                    cv::Point p1 = db.feturePoints[j];
                    cv::Point p2 = db.feturePoints[(j + 1)]; // 다음 점, 마지막 점에서 첫 점으로 순환
                    pairs.emplace_back(p1, p2);
                }  
            }

            std::cout << std::endl;
            for (size_t t=0; t<pairs.size(); t++)
            {
                std::cout <<"("<<pairs[t].p1.x << ", " << pairs[t].p1.y << ") - ("<<pairs[t].p2.x << ", " << pairs[t].p2.y 
                << ") dist: " << pairs[t].dist << endl; 

                if (pairs[t].dist < 20)
                {
                    cv::line(img_freeSpace, cv::Point(pairs[t].p1), cv::Point(pairs[t].p2), cv::Scalar(0), 2);
                    cv::line(result_img, cv::Point(pairs[t].p1), cv::Point(pairs[t].p2), cv::Scalar(0, 0, 255), 2);   
                } 
            }  

            std::cout <<std::endl;
            std::cout << "TrajectoryPts:";
            for (const auto& pt : db.trajectoryPoints) {

                cv::circle(result_img, pt, 1, cv::Scalar(255, 0, 0), -1);
                std::cout << "(" << pt.x << ", " << pt.y << ") ";
            };

            std::cout <<std::endl;
            std::cout <<"-----------------------------------------------------" <<std::endl;   
            ++i; 
        }
        cv::imshow("result_img2", result_img);
        cv::imshow("img_freeSpace2", img_freeSpace);
        
        cv::waitKey();

    }
    


    cv::Mat dst;
    double scaleFactor = 1.5;
    cv::resize(result_img, dst, cv::Size(), scaleFactor, scaleFactor);
    cv::imshow("dst", dst);


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

