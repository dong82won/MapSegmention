
#include "utility.h"

// Custom comp
cv::Scalar randomColor()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 255);

    return cv::Scalar(dist(gen), dist(gen), dist(gen)); // 임의의 BGR 색상 생성
}

std::vector<cv::Point> changeMatoPoint(cv::Mat &image)
{
    std::vector<cv::Point> edgePts;
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            if (image.at<uchar>(i, j) == 255)
            {
                edgePts.push_back(cv::Point(j, i));
            }
        }
    }
    return edgePts;
}

cv::Mat extractMat2Mat(cv::Mat &image, uchar pixel_value)
{
    cv::Mat dst=cv::Mat::zeros(image.size(), CV_8UC1);
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            if (image.at<uchar>(i, j) == pixel_value)
            {
                dst.at<uchar>(i, j) = 255;
            }
        }
    }
    return dst;
}




// 거리 계산 함수
double calculateDistance(const cv::Point &p1, const cv::Point &p2)
{
    return sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2));
}

// 점들을 순차적으로 정렬하여 실선 재구성
std::vector<cv::Point> sortPoints(const std::vector<cv::Point> &points)
{
    std::vector<cv::Point> sortedPoints;

    if (points.empty())
        return sortedPoints;

    std::vector<cv::Point> remainingPoints = points;
    cv::Point currentPoint = remainingPoints[0];

    sortedPoints.push_back(currentPoint);
    remainingPoints.erase(remainingPoints.begin());

    while (!remainingPoints.empty())
    {
        auto nearestIt = std::min_element(remainingPoints.begin(), remainingPoints.end(),
                                          [&currentPoint](const cv::Point &p1, const cv::Point &p2)
                                          {
                                              return calculateDistance(currentPoint, p1) < calculateDistance(currentPoint, p2);
                                          });
        currentPoint = *nearestIt;
        sortedPoints.push_back(currentPoint);
        remainingPoints.erase(nearestIt);
    }

    return sortedPoints;
}

// 중복되는 cv::Point 데이터를 제거하는 함수
void removeDuplicatePoints(std::vector<cv::Point> &points)
{
    // points를 정렬
    std::sort(points.begin(), points.end(), [](const cv::Point &a, const cv::Point &b)
              { return (a.x < b.x) || (a.x == b.x && a.y < b.y); });

    // 중복된 요소를 points의 끝으로 이동
    auto last = std::unique(points.begin(), points.end());

    // 중복된 요소를 제거
    points.erase(last, points.end());
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

void drawingSetpRectangle(cv::Mat &image, cv::Point circlesCenters, int radius)
{
    int max_x = circlesCenters.x + radius;
    int max_y = circlesCenters.y + radius;
    int min_x = circlesCenters.x - radius;
    int min_y = circlesCenters.y - radius;
    cv::Rect rect(cv::Point(min_x, min_y), cv::Point(max_x, max_y));

    cv::rectangle(image, rect, cv::Scalar(255, 0, 0), 1); // 파란색 사각형
}


// 주어진 중심과 반지름을 기반으로 원의 경계 점을 찾는 함수
std::vector<cv::Point> edgePointsInCircle(const cv::Point &center, int radius)
{
    std::vector<cv::Point> points;

    // 원의 경계에서 점들을 추출
    for (double angle = 0; angle < 2 * CV_PI; angle += 0.1)
    { // 각도를 0.1씩 증가시켜 점을 추출
        int x = static_cast<int>(center.x + radius * cos(angle));
        int y = static_cast<int>(center.y + radius * sin(angle));
        points.push_back(cv::Point(x, y));
    }

    return points;
}

void drawingOutLineCircule(const cv::Mat &image, cv::Point circlesCenters, int radius)
{
     // 원형 영역을 이미지에 그리기
    std::vector<cv::Point> edgePoints = edgePointsInCircle(circlesCenters, radius);
    
    // 원을 실선으로 그리기
    for (size_t i = 0; i < edgePoints.size(); i++)
    {
        cv::Point start = edgePoints[i];
        cv::Point end = edgePoints[(i + 1) % edgePoints.size()]; // 마지막 점과 첫 점을 연결
        cv::line(image, start, end, cv::Scalar(255, 0, 0), 1);       // 파란색으로 실선 그리기
    }
}

