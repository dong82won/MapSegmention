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

    // 생성자
    SEGDATA() = default;    
    SEGDATA(const cv::Point& key, const std::vector<cv::Point>& feture, const std::vector<cv::Point>& traj)
        : centerPoint(key), feturePoints(feture), trajectoryPoints(traj) {}

    // 경로에 포인트 추가
    void addFeturePoints(const cv::Point& point) {
        feturePoints.push_back(point);
    }

    void addTrajectoryPoint(const cv::Point& point) {
        trajectoryPoints.push_back(point);
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

int main() {
    std::string home_path = getenv("HOME");
    // std::cout << home_path << std::endl;

    // 이미지 파일 경로
    cv::Mat raw_img = cv::imread(home_path + "/myStudyCode/MapSegmention/imgdb/occupancy_grid.png", cv::IMREAD_GRAYSCALE);
    //cv::Mat raw_img = cv::imread(home_path + "/myWorkCode/regonSeg/imgdb/caffe_map.pgm", cv::IMREAD_GRAYSCALE);
    if (raw_img.empty())
    {
        std::cerr << "Error: Unable to open image file: " << std::endl;
        return -1;
    }

    cv::Mat result_img;
    cv::cvtColor(raw_img, result_img, cv::COLOR_GRAY2RGB);
    cv::Mat result_img2 = result_img.clone();

    FeatureDetection fd(raw_img);
    fd.straightLineDetection();

    // 병합 픽셀 설정: 9, 12;
    fd.detectEndPoints(9);
    std::vector<cv::Point> fpoints = fd.getUpdateFeaturePoints();
 

    for (size_t i = 0; i< fpoints.size(); i++)
    {
        cv::Point pt = fpoints[i];;
        cv::circle(result_img, pt, 3, CV_RGB(0, 255, 0), -1); 
    }
    
    cv::imshow("result_img", result_img);

    
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
    
    for (const auto &pt : sorted_trajectory_points)
    {
        cv::circle(result_img, pt, 1, CV_RGB(255, 0, 0), -1);
    }

    int radius = 15; // 탐색 범위 반지름
    vector<Point> circlesCenters = addHalfOverlappingCircles(sorted_trajectory_points, radius);

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



        for (size_t i = 0; i < circlesCenters.size(); i++)
        {
            cv::Point cp = circlesCenters[i];
            SEGDATA db = testExploreFeature3( fpoints, sorted_trajectory_points, cp, radius);
                        
            std::cout <<"-----------------------------------------------------" <<std::endl;
            std::cout << "Key Point: (" << db.centerPoint.x << ", " << db.centerPoint.y << ")\n";            
            std::cout << "         FeaturePts:";
            for (const auto& pt : db.feturePoints) {
                std::cout << "(" << pt.x << ", " << pt.y << ") ";
            };            
            std::cout <<std::endl;
            std::cout << "         TrajectoryPts:";
            for (const auto& pt : db.trajectoryPoints) {
                std::cout << "(" << pt.x << ", " << pt.y << ") ";
            };
            std::cout <<std::endl;
            std::cout <<"-----------------------------------------------------" <<std::endl;


        // for (const auto &pair : db)
        // {
        //     const cv::Point &key = pair.first;
        //     const std::vector<cv::Point> &value = pair.second;

        //     if (value.size() > 1)
        //     {
        //         drawingSetpCircule(result_img, cp, radius);
        //         std::cout << "Key Point: (" << key.x << ", " << key.y << ")\n";
        //         std::cout << "Values: ";
        //         for (const auto &pt : value)
        //         {
        //             std::cout << "(" << pt.x << ", " << pt.y << ") ";
        //         }
        //         std::cout << std::endl;
        //     }
        // }

            cv::imshow("result_img2", result_img);
            cv::waitKey();
        }

        return 0;
    }

    //     // 결과 외곽선을 vector<Point>에 저장
    // vector<Point> result_contour; 
    // // 그린 외곽선의 좌표를 추출하여 저장
    // for (int y = 0; y < img_freeSpace.rows; y++) {
    //     for (int x = 0; x < img_freeSpace.cols; x++) {
    //         uchar pixel = img_freeSpace.at<uchar>(y, x);
    //         // 만약 픽셀이 초록색 (0, 255, 0)이라면
    //         if (pixel == 128) {
    //             result_contour.push_back(Point(x, y));
    //         }
    //     }
    // }

    // //Mat temp_img = Mat::zeros(img_freeSpace.size(), CV_8UC1);
    // Mat temp_img(img_freeSpace.size(), CV_8UC1, Scalar(255));

    // for (const auto& pt : result_contour) {
    //     circle(temp_img, pt, 1, Scalar(0), -1);
    // }
    // cv::imshow("temp_img", temp_img);

    // ---------------------------------------------------------------------

    // std::vector<cv::Point> sort_map_contours = sortPoints(map_contours);

    // for (const auto &pt : map_contours)
    // {
    //     cv::circle(color_img_freeSpace, pt, 1, CV_RGB(0, 255, 0), -1);
    // }

    // for (size_t i = 0; i < map_contours.size(); ++i) 
    // {
    //     for (size_t j = 0; j < map_contours[i].size()-1; ++j)
    //     {
    //         Point pt1 = map_contours[i][j];
    //         Point pt2 = map_contours[i][j+1];
    //         line(color_img_freeSpace, pt1, pt2, Scalar(0, 255, 0), 2);
    //     }
    // }



    // cv::imshow("color_img_freeSpace", color_img_freeSpace);




//------------------------------------------------------------------------------

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

