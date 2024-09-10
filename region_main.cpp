// 집에서  9/5 10:13

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <stack>
#include <algorithm>

int main()
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

//     // 이미지 파일 경로
//     cv::Mat raw_img = cv::imread(home_path + "/myStudyCode/MapSegmention/imgdb/occupancy_grid.png", cv::IMREAD_GRAYSCALE);
//     // cv::Mat raw_img = cv::imread(home_path + "/myWorkCode/regonSeg/imgdb/caffe_map.pgm", cv::IMREAD_GRAYSCALE);
//     if (raw_img.empty())
//     {
//         std::cerr << "Error: Unable to open image file: " << std::endl;
//         return -1;
//     }
//     cv::Mat result_img;
//     cv::cvtColor(raw_img, result_img, cv::COLOR_GRAY2RGB);
//     cv::Mat result_img2 = result_img.clone();

// //     FeatureDetection fd(raw_img);
// //     fd.straightLineDetection();

// //     // 병합 픽셀 설정: 9, 12;
// //     fd.detectEndPoints(9);
// //     // fd.getUpdateFeaturePoints();

// //     for (const auto &pt : fd.getUpdateFeaturePoints())
// //     {
// //         cv::circle(result_img, pt, 3, cv::Scalar(0, 0, 255), -1);
// //     }
    

//     cv::Mat img_freeSpace = makeFreeSpace(raw_img);
//     imshow("img_freeSpace", img_freeSpace);

        
//     // 윤곽선 찾기
//     vector<vector<Point>> contours;
//     vector<Vec4i> hierarchy;
//     findContours(img_freeSpace, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);

//     // 컨투어 검출
//     findContours(img_freeSpace, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

//     // 검출된 컨투어를 원본 이미지에 그리기
//     Mat drawing_contours = Mat::zeros(img_freeSpace.size(), CV_8UC1);    
//     for (size_t i = 0; i < contours.size(); i++)
//     {
//         // cout << "contours.size(): "<< contours[i].size() << endl;
//         if (contours[i].size() > 15)
//             drawContours(drawing_contours, contours, (int)i, Scalar(255), 1);
//     }
//     imshow("drawing_contours", drawing_contours);


//     vector<Point> contour;
//     for (int i = 0; i < drawing_contours.rows; i++)
//     {
//         for (int j= 0; j < drawing_contours.cols; j++)
//         {
//             if (drawing_contours.at<uchar>(i, j) == 255) 
//                 contour.push_back(Point(j, i));
//         }
//     }


//     for (const auto &pt : contour)
//     {
//         cv::circle(result_img, pt, 3, cv::Scalar(0, 255, 0), -1);        
//     }
//     imshow("result_img", result_img);





    // // cv::circle(result_img, seedPoint, 3, cv::Scalar(255, 255, 0), -1);
    // // detectEndPoints(drawing, result_contour);

    // for (const auto &pt : result_contour)
    // {
    //     cv::circle(result_img, pt, 3, cv::Scalar(0, 255, 0), -1);
    // }
    // imshow("test", result_img);

    // // 다각형 채우기
    // floodFill(drawing, Point(img_freeSpace.cols / 2, img_freeSpace.rows / 2), Scalar(255));

    // // 결과 이미지 출력
    // imshow("Drawing_Region", drawing);


    // //-----------------------------------------------------
    // TrajectionPoint tp;
    // cv::Mat img_dist = tp.makeDistanceTransform(drawing);
    // imshow("img_dist", img_dist);

    // cv::Mat img_skeletion;
    // tp.zhangSuenThinning(img_dist, img_skeletion); 
    // cv::imshow("img_skeletion", img_skeletion);      



    Mat image = Mat::zeros(500, 500, CV_8UC3);

    // 데이터로 사용할 점들 설정
    vector<Point> data = {
        Point(100, 100), Point(110, 110), Point(120, 120), 
        Point(130, 130), Point(140, 140), Point(150, 150)
    };

    int radius = 30; // 탐색 범위 반지름

    // 원형 탐색 범위가 겹치지 않도록 점들을 추가
    vector<Point> circlesCenters = addNonOverlappingCircles(data, radius);

    // 원형 영역을 이미지에 채우기
    for (const auto& center : circlesCenters) {
        fillCircle(image, center, radius, Scalar(0, 255, 0)); // 초록색으로 원 내부 채우기
    }

    drawingOutLineCircule(image, circlesCenters, radius);

    // 데이터 포인트를 이미지에 표시
    for (const auto& point : data) {
        fillCircle(image, point, 3, Scalar(0, 0, 255)); // 빨간색 점 표시
    }



    // // 원형 영역을 이미지에 그리기
    // for (const auto& center : circlesCenters) {
    //     vector<Point> edgePoints = edgePointsInCircle(center, radius);
        
    //     // 원의 경계 점을 초록색으로 표시
    //     for (const auto& point : edgePoints) {
    //         if (point.x >= 0 && point.x < image.cols && point.y >= 0 && point.y < image.rows) {
    //             image.at<Vec3b>(point.y, point.x) = Vec3b(0, 255, 0); 
    //             //circle(image, point, 3, Scalar(0, 0, 255), -1); // 빨간색 점 표시                
    //         }
    //     }        
    //     // // 원을 실선으로 그리기
    //     // for (size_t i = 0; i < edgePoints.size(); i++) {
    //     //     Point start = edgePoints[i];
    //     //     Point end = edgePoints[(i + 1) % edgePoints.size()]; // 마지막 점과 첫 점을 연결
    //     //     line(image, start, end, Scalar(255, 0, 0), 1); // 파란색으로 실선 그리기
    //     // }
    // }

    // // 데이터 포인트를 이미지에 표시
    // for (const auto& point : data) {
    //     circle(image, point, 3, Scalar(0, 0, 255), -1); // 빨간색 점 표시
    // }

    imshow("image", image);
    waitKey();


    return 0;
}
