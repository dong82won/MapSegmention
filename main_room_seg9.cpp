#include <opencv2/opencv.hpp>

#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <unordered_set>

#include <algorithm>

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

/**/
// Zhang-Suen Thinning Algorithm
void zhangSuenThinning(const cv::Mat &src, cv::Mat &dst)
{

    cv::Mat img;
    int th = (int)cv::threshold(src, img, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    img /= 255;
    cv::Mat prev = cv::Mat::zeros(img.size(), CV_8UC1);
    cv::Mat diff;

    do
    {
        cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

        for (int y = 1; y < img.rows - 1; ++y)
        {
            for (int x = 1; x < img.cols - 1; ++x)
            {
                uchar p2 = img.at<uchar>(y - 1, x);
                uchar p3 = img.at<uchar>(y - 1, x + 1);
                uchar p4 = img.at<uchar>(y, x + 1);
                uchar p5 = img.at<uchar>(y + 1, x + 1);
                uchar p6 = img.at<uchar>(y + 1, x);
                uchar p7 = img.at<uchar>(y + 1, x - 1);
                uchar p8 = img.at<uchar>(y, x - 1);
                uchar p9 = img.at<uchar>(y - 1, x - 1);

                int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                        (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                        (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                        (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                if (img.at<uchar>(y, x) == 1 && B >= 2 && B <= 6 && A == 1 &&
                    (p2 * p4 * p6 == 0) && (p4 * p6 * p8 == 0))
                {
                    marker.at<uchar>(y, x) = 1;
                }
            }
        }

        img -= marker;

        for (int y = 1; y < img.rows - 1; ++y)
        {
            for (int x = 1; x < img.cols - 1; ++x)
            {
                uchar p2 = img.at<uchar>(y - 1, x);
                uchar p3 = img.at<uchar>(y - 1, x + 1);
                uchar p4 = img.at<uchar>(y, x + 1);
                uchar p5 = img.at<uchar>(y + 1, x + 1);
                uchar p6 = img.at<uchar>(y + 1, x);
                uchar p7 = img.at<uchar>(y + 1, x - 1);
                uchar p8 = img.at<uchar>(y, x - 1);
                uchar p9 = img.at<uchar>(y - 1, x - 1);

                int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                        (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                        (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                        (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                if (img.at<uchar>(y, x) == 1 && B >= 2 && B <= 6 && A == 1 &&
                    (p2 * p4 * p8 == 0) && (p2 * p6 * p8 == 0))
                {
                    marker.at<uchar>(y, x) = 1;
                }
            }
        }

        img -= marker;
        cv::absdiff(img, prev, diff);
        img.copyTo(prev);

    } while (cv::countNonZero(diff) > 0);

    img *= 255;
    dst = img.clone();
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

// // 체인코드를 생성하는 함수
// vector<int> generateChainCode(const vector<Point>& contour) {
    
//     // 8방향 체인코드를 위한 방향 배열
//     const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
//     const int dy[8] = {0, -1, -1, -1, 0, 1, 1, 1};
    
//     vector<int> chainCode;

//     for (size_t i = 0; i < contour.size() - 1; ++i) {
//         Point current = contour[i];
//         Point next = contour[i + 1];
        
//         // 다음 픽셀로의 방향 계산
//         int dx_val = next.x - current.x;
//         int dy_val = next.y - current.y;
        
//         // 방향을 체인코드로 변환
//         for (int direction = 0; direction < 8; ++direction) {
//             if (dx[direction] == dx_val && dy[direction] == dy_val) {
//                 chainCode.push_back(direction);
//                 break;
//             }
//         }
//     }

//     return chainCode;
// }
 // 연결된 성분을 찾는 함수
void findConnectedComponents(const vector<Point>& contour, vector<vector<Point>>& components) {
    

    // 체인 코드 방향 (8방향)
    int dx[] = {1, 1, 0, -1, -1, -1, 0, 1};
    int dy[] = {0, 1, 1, 1, 0, -1, -1, -1};
 
    vector<bool> visited(contour.size(), false);
    
    // 현재 성분을 저장할 변수
    vector<Point> currentComponent;

    for (size_t i = 0; i < contour.size(); i++) {
        if (visited[i]) continue; // 이미 방문한 점은 건너뜀

        // 새로운 성분 시작
        currentComponent.clear();
        vector<Point> stack; // DFS를 위한 스택
        stack.push_back(contour[i]);

        while (!stack.empty()) {
            Point p = stack.back();
            stack.pop_back();

            // 현재 점을 성분에 추가하고 방문 처리
            currentComponent.push_back(p);
            int idx = find(contour.begin(), contour.end(), p) - contour.begin();
            visited[idx] = true;

            // 8방향으로 탐색
            for (int dir = 0; dir < 8; dir++) {
                Point neighbor(p.x + dx[dir], p.y + dy[dir]);

                // 인접 점이 윤곽선에 존재하고 방문하지 않았을 경우 스택에 추가
                if (find(contour.begin(), contour.end(), neighbor) != contour.end() && !visited[find(contour.begin(), contour.end(), neighbor) - contour.begin()]) {
                    stack.push_back(neighbor);
                }
            }
        }

        // 연결된 성분을 저장
        components.push_back(currentComponent);
    }
}

// 함수: 가장 긴 직선 찾기
Vec4i findLongestLine(const vector<Vec4i>& lines) {
    Vec4i longestLine;
    double maxLength = 0;

    for (const auto& line : lines) {
        double length = norm(Point(line[0], line[1]) - Point(line[2], line[3]));
        if (length > maxLength) {
            maxLength = length;
            longestLine = line;
        }
    }

    return longestLine;
}

// 함수: 이미지 회전
Mat rotateImage(const Mat& image, double angle) {
    Point2f center(image.cols / 2.0, image.rows / 2.0);
    Mat rotationMatrix = getRotationMatrix2D(center, angle, 1.0);
    Mat rotatedImage;
    warpAffine(image, rotatedImage, rotationMatrix, image.size());
    return rotatedImage;
}

 


// void gridSnapping(const Mat& inputImage, Mat& outputImage, int gridSize) {
//     outputImage = Mat::zeros(inputImage.size(), inputImage.type()); // 결과 이미지를 0으로 초기화

//     // 이미지에서 모든 픽셀을 순회
//     for (int y = 0; y < inputImage.rows; y++) {
//         for (int x = 0; x < inputImage.cols; x++) {
//             // 흰색 픽셀(255)인 경우
//             if (inputImage.at<uchar>(y, x) == 255) {
//                 // 가장 가까운 격자 점 계산
//                 int snappedX = round(static_cast<double>(x) / gridSize) * gridSize;
//                 int snappedY = round(static_cast<double>(y) / gridSize) * gridSize;

//                 // 격자 점으로 픽셀 이동
//                 if (snappedX >= 0 && snappedX < outputImage.cols && snappedY >= 0 && snappedY < outputImage.rows) {
//                     outputImage.at<uchar>(snappedY, snappedX) = 255; // 흰색 픽셀로 설정
//                 }
//             }
//         }
//     }
// }

// 격자 점 계산 함수
Point calculateSnappedPoint(const Point& pixel, int gridSize) {
    int snappedX = round(static_cast<double>(pixel.x) / gridSize) * gridSize;
    int snappedY = round(static_cast<double>(pixel.y) / gridSize) * gridSize;
    return Point(snappedX, snappedY);
}

void gridSnapping(const Mat& inputImage, Mat& outputImage, int gridSize) {
    outputImage = Mat::zeros(inputImage.size(), inputImage.type()); // 결과 이미지를 0으로 초기화

    // 이미지에서 모든 픽셀을 순회
    for (int y = 0; y < inputImage.rows; y++) {
        for (int x = 0; x < inputImage.cols; x++) {
            // 흰색 픽셀(255)인 경우
            if (inputImage.at<uchar>(y, x) == 255) {
                // 가장 가까운 격자 점 계산
                Point snappedPoint = calculateSnappedPoint(Point(x, y), gridSize);

                // 격자 점으로 픽셀 이동
                int halfGridSize = (gridSize / 2)+1;

                // 지정된 격자 크기만큼의 영역을 흰색으로 설정
                for (int dy = -halfGridSize; dy < halfGridSize; dy++) {
                    for (int dx = -halfGridSize; dx < halfGridSize; dx++) {
                        int newY = snappedPoint.y + dy;
                        int newX = snappedPoint.x + dx;

                        // 이미지 경계 내에서만 설정
                        if (newX >= 0 && newX < outputImage.cols && 
                            newY >= 0 && newY < outputImage.rows) {
                            outputImage.at<uchar>(newY, newX) = 255; // 흰색 픽셀로 설정
                        }
                    }
                }
            }
        }
    }
}


std::vector<cv::Point> makeImagetoPoint(cv::Mat& image)
{

    std::vector<cv::Point> edgePts; 
    for (int i=0; i<image.rows; i++)
    {
        for (int j=0; j<image.cols; j++)
        {
            if (image.at<uchar>(i,j) == 255)
            {
                edgePts.push_back(cv::Point(j, i)); 
            }
        }
    }
    return edgePts; 
}

// 거리 계산 함수
double calculateDistance(const Point &p1, const Point &p2)
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


// 두 원형 영역이 반만 겹치는지 확인하는 함수
bool isHalfOverlap(const Point& center1, int radius1, const Point& center2, int radius2) {
    double distance = sqrt(pow(center1.x - center2.x, 2) + pow(center1.y - center2.y, 2));
    return distance <= (radius1 + radius2) / 2.0;
}

// 원형 탐색 범위를 추가하는 함수
vector<Point> addHalfOverlappingCircles(const vector<Point> &data, int radius)
{
    vector<Point> circlesCenters;

    for (const auto &point : data)
    {
        bool overlap = false;

        // 새로 추가할 원형 범위가 기존의 범위와 반만 겹치는지 확인
        for (const auto &existingCenter : circlesCenters)
        {
            if (isHalfOverlap(existingCenter, radius, point, radius))
            {
                overlap = true;
                break;
            }
        }

        if (!overlap)
        {
            circlesCenters.push_back(point);
        }
    }

    return circlesCenters;
}


struct LINEINFO 
{
    std::pair<cv::Point, cv::Point> virtual_wll;
    double distance; 

      // 두 LINEINFO가 동일한지 비교하는 연산자
    bool operator==(const LINEINFO& other) const {
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

    //std::vector<LINEINFO> lines;
    
    //bool state_ = false;

    // 생성자
    SEGDATA() = default;
    SEGDATA(const cv::Point &key, const std::vector<cv::Point> &feture, const std::vector<cv::Point> &traj)
        : centerPoint(key), feturePoints(feture), trajectoryPoints(traj) {}

    // void addState(bool state)
    // {
    //     state_ = state;
    // }

};


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


void drawingSetpRectangle(Mat &image, cv::Point circlesCenters, int radius)
{
    int max_x = circlesCenters.x + radius;
    int max_y = circlesCenters.y + radius;
    int min_x = circlesCenters.x - radius;
    int min_y = circlesCenters.y - radius;
    cv::Rect rect(cv::Point(min_x, min_y), cv::Point(max_x, max_y));

    cv::rectangle(image, rect, cv::Scalar(255, 0, 0), 1); // 파란색 사각형
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



//-----------------------------------------------------------------------------------------------------
// 선분과 점 사이의 수직 거리를 계산하는 함수
double pointToLineDistance(const cv::Point &p, const cv::Point &lineStart, const cv::Point &lineEnd)
{
    double A = p.x - lineStart.x;
    double B = p.y - lineStart.y;
    double C = lineEnd.x - lineStart.x;
    double D = lineEnd.y - lineStart.y;

    double dot = A * C + B * D;
    double len_sq = C * C + D * D;
    double param = (len_sq != 0) ? dot / len_sq : -1; // 선분의 길이가 0이 아닐 때만 계산

    double xx, yy;

    if (param < 0)
    {
        xx = lineStart.x;
        yy = lineStart.y;
    }
    else if (param > 1)
    {
        xx = lineEnd.x;
        yy = lineEnd.y;
    }
    else
    {
        xx = lineStart.x + param * C;
        yy = lineStart.y + param * D;
    }

    double dx = p.x - xx;
    double dy = p.y - yy;
    return std::sqrt(dx * dx + dy * dy);
}

// 직선 세그먼트 근처에 점이 있는지 확인하는 함수
bool isPointNearLine(const cv::Point &p, const cv::Point &lineStart, const cv::Point &lineEnd, double threshold)
{
    // 점이 선분에서 특정 거리 이내에 있는지 확인
    double distance = pointToLineDistance(p, lineStart, lineEnd);
    return distance <= threshold;
}


// 데이터 A와 B에 대해 직선 세그먼트에서 점을 확인하는 함수
std::vector<LINEINFO> checkPointsNearLineSegments(const std::vector<cv::Point> &dataA, const std::vector<cv::Point> &dataB, double distance_threshold = 5.0)
{
    
    LINEINFO line;
    std::vector<LINEINFO> lines;
    
    for (size_t i = 0; i < dataA.size(); ++i)
    {
        for (size_t j = i + 1; j < dataA.size(); ++j)
        {
            cv::Point start = dataA[i];
            cv::Point end = dataA[j];

            std::cout << "Line segment: (" << start.x << ", " << start.y << ") -> ("
                      << end.x << ", " << end.y << ") = distance "<<  calculateDistance(start, end)  << "\n";

            bool foundPointNearLine = false;

            for (const auto &bPoint : dataB)
            {
                if (isPointNearLine(bPoint, start, end, distance_threshold))
                {
                    std::cout << "    Point near line: (" << bPoint.x << ", " << bPoint.y << ")\n";
                    foundPointNearLine = true;
                }
            }

            if (foundPointNearLine)
            {   
                line.virtual_wll = std::make_pair(start, end);
                line.distance = calculateDistance(start, end); 
                lines.emplace_back(line);

            }
            else {
                std::cout << "    No points from dataB are near this line.\n";
            }
        }
    }

    return lines;
}


// Custom comparator for cv::Point
struct PointComparator {
    bool operator()(const cv::Point& a, const cv::Point& b) const {
        return (a.x < b.x) || (a.x == b.x && a.y < b.y);
    }
};


// Custom comparator for LINEINFO to use in a set
struct LineInfoComparator {
    bool operator()(const LINEINFO& a, const LINEINFO& b) const {
        return (PointComparator{}(a.virtual_wll.first, b.virtual_wll.first)) || 
               (a.virtual_wll.first == b.virtual_wll.first && PointComparator{}(a.virtual_wll.second, b.virtual_wll.second));
    }
};

// Function to check if two LINEINFO objects are equal regardless of the order of points
bool areEqualIgnoringOrder(const LINEINFO& a, const LINEINFO& b) {
    return (a.virtual_wll == b.virtual_wll) || 
           (a.virtual_wll.first == b.virtual_wll.second && a.virtual_wll.second == b.virtual_wll.first);
}

//Function to remove duplicates from a vector of LINEINFO
std::vector<LINEINFO> removeDuplicates(const std::vector<LINEINFO>& lines) {
    std::set<LINEINFO, LineInfoComparator> uniqueLines;

    for (const auto& line : lines) {
        bool isDuplicate = false;
        for (const auto& uniqueLine : uniqueLines) {
            if (areEqualIgnoringOrder(line, uniqueLine)) {
                isDuplicate = true;
                break;
            }
        }
        if (!isDuplicate) {
            uniqueLines.insert(line);
        }
    }

    // Convert back to vector
    return std::vector<LINEINFO>(uniqueLines.begin(), uniqueLines.end());
}

// 커스텀 해시 함수 (좌표와 거리로 해시 값을 생성)
struct LineInfoHasher {
    size_t operator()(const LINEINFO& line) const {
        auto hash1 = std::hash<int>()(line.virtual_wll.first.x) ^ std::hash<int>()(line.virtual_wll.first.y);
        auto hash2 = std::hash<int>()(line.virtual_wll.second.x) ^ std::hash<int>()(line.virtual_wll.second.y);
        auto hash3 = std::hash<double>()(line.distance);
        return hash1 ^ hash2 ^ hash3; // 좌표와 거리의 해시값을 결합
    }
};

// 두 선이 끝점을 기준으로 겹치는지 확인하는 함수
bool linesOverlap(const LINEINFO& line1, const LINEINFO& line2) {
    return (line1.virtual_wll.first == line2.virtual_wll.first || 
            line1.virtual_wll.first == line2.virtual_wll.second ||
            line1.virtual_wll.second == line2.virtual_wll.first || 
            line1.virtual_wll.second == line2.virtual_wll.second);
}

// 선들을 겹침 조건에 따라 필터링하는 함수 (동일한 선은 하나만 남김)
std::vector<LINEINFO> filterLines(std::vector<LINEINFO>& lines) {
    std::unordered_set<LINEINFO, LineInfoHasher> uniqueLines; // 중복을 제거할 해시 셋
    std::vector<LINEINFO> result; // 결과를 저장할 벡터

    for (size_t i = 0; i < lines.size(); ++i) {
        bool toRemove = false;

        // 중복 체크: 이미 처리된 동일한 선이 있는지 확인
        if (uniqueLines.find(lines[i]) != uniqueLines.end()) {
            continue; // 동일한 선이 이미 있으면 건너뜀
        }
        
        for (size_t j = 0; j < lines.size(); ++j) {
            if (i != j && linesOverlap(lines[i], lines[j])) {
                if (lines[i].distance > lines[j].distance) {
                    toRemove = true;
                    break;
                }
            }
        }

        if (!toRemove) {
            uniqueLines.insert(lines[i]); // 중복되지 않으면 셋에 삽입
            result.push_back(lines[i]);   // 결과에 추가
        }
    }

    return result;
}
// std::vector<std::vector<LINEINFO>>를 std::vector<LINEINFO>로 변환하는 함수
std::vector<LINEINFO> convertToLineInfo(const std::vector<std::vector<LINEINFO>>& a) {
    std::vector<LINEINFO> b; // 1D 벡터를 위한 벡터

    // 2D 벡터를 순회하며 각 요소를 1D 벡터에 추가
    for (const auto& innerVector : a) {
        b.insert(b.end(), innerVector.begin(), innerVector.end());
    }

    return b; // 변환된 벡터 반환
}

// 윤곽선의 크기 조절 함수
std::vector<cv::Point> scaleContour(const std::vector<cv::Point>& contour, double scaleFactor) {
    std::vector<cv::Point> scaledContour;
    for (const auto& pt : contour) {
        scaledContour.push_back(cv::Point(pt.x * scaleFactor, pt.y * scaleFactor));
    }
    return scaledContour;
}


int main()
{
    std::string home_path = getenv("HOME");
    // std::cout << home_path << std::endl;

    // 이미지 파일 경로
    cv::Mat raw_img = cv::imread(home_path + "/myStudyCode/MapSegmention/imgdb/occupancy_grid.png", cv::IMREAD_GRAYSCALE);
    // cv::Mat wall_img = cv::imread(home_path + "/myWorkCode/MapSegmention/imgdb/occupancy_grid_wall.png", cv::IMREAD_GRAYSCALE);
    // cv::imshow("wall_img", wall_img);
    // cv::Mat raw_img = cv::imread(home_path + "/myWorkCode/regonSeg/imgdb/caffe_map.pgm", cv::IMREAD_GRAYSCALE);
    if (raw_img.empty())
    {
        std::cerr << "Error: Unable to open image file: " << std::endl;
        return -1;
    } 
    imshow("raw_img",raw_img);
    cv::Mat img_wall = makeImageWall(raw_img); //(occupancyMap);
    imshow("occupancyMap", img_wall);
 

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat img_open;
    morphologyEx(img_wall, img_open, MORPH_DILATE, kernel, Point(-1, -1), 1); 
    imshow("img_open", img_open);

    
    // 직선 검출 (Hough 변환)
    vector<Vec4i> lines;
    HoughLinesP(img_open, lines, 1, CV_PI / 180, 100, 50, 10); 
    // 가장 긴 직선 찾기
    Vec4i longestLine = findLongestLine(lines); 
    // 직선의 기울기 계산
    double angle = atan2(longestLine[3] - longestLine[1], longestLine[2] - longestLine[0]) * 180.0 / CV_PI; 
    
    // 이미지 회전
    Mat rotatedImage = rotateImage(img_open, angle);
    Mat rotated_raw_img = rotateImage(raw_img, angle);

    cv::imshow("rotatedImage", rotatedImage);
    cv::imshow("rotated_raw_img", rotated_raw_img);

    cv::Mat color_rotated_raw_img; 
    cv::cvtColor(rotated_raw_img, color_rotated_raw_img, COLOR_GRAY2BGR);
    
    cv::Mat test_rotated_img = color_rotated_raw_img.clone();
    cv::Mat test_rotated_img2 = color_rotated_raw_img.clone();
    
    cv::Mat img_wall_skeletion;
    zhangSuenThinning(rotatedImage, img_wall_skeletion);
    cv::imshow("img_wall_skeletion", img_wall_skeletion);

    // std::vector<cv::Point> edgePts; 
    // for (int i=0; i<img_wall_skeletion.rows; i++)
    // {
    //     for (int j=0; j<img_wall_skeletion.cols; j++)
    //     {
    //         if (img_wall_skeletion.at<uchar>(i,j) == 255)
    //         {
    //             edgePts.push_back(cv::Point(j, i)); 
    //         }
    //     }
    // }

    std::vector<cv::Point> edgePts = makeImagetoPoint(img_wall_skeletion);

    cv::Mat img_test= cv::Mat::zeros(img_wall_skeletion.size(), CV_8UC3);

    // for (const auto &pt: edgePts)
    // {
    //     cv::circle(img_test, pt, 3, cv::Scalar(0, 255, 0), -1);
    // } 
    // cv::imshow("img_test", img_test); 
    

    cv::Mat img_edge= cv::Mat::zeros(img_wall_skeletion.size(), CV_8UC1);
    // 연결된 성분을 저장할 벡터
    vector<vector<Point>> segEdgePts;  
    findConnectedComponents(edgePts,  segEdgePts); 
  
    cout << "segEdgePts.size(): " <<segEdgePts.size() << endl;
    for (size_t i =0; i<segEdgePts.size(); i++)
    {
        cout<<"edge_num: " << segEdgePts[i].size() << endl;
        if (segEdgePts[i].size()> 20)
        {
            for (const auto &pt: segEdgePts[i])
            {  
                cv::circle(img_test, pt, 0, cv::Scalar(0, 255, 0), -1);

                int y = pt.y;
                int x = pt.x;
                img_edge.at<uchar>(y, x) = 255; 
            }
    
        } 
    }

    cv::imshow("img_test", img_test); 
    cv::imshow("img_edge", img_edge);


    Mat img_grid;
    int gridSize = 3; // 격자 크기 설정
    gridSnapping(img_edge, img_grid, gridSize);
    imshow("gridSnapping", img_grid);





    
    cv::Mat img_grid_skeletion;
    zhangSuenThinning(img_grid, img_grid_skeletion);
    cv::imshow("img_grid_skeletion", img_grid_skeletion);

    // 수평선 감지
    cv::Mat horizontal;
    cv::Mat horizontal_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 1));
    cv::morphologyEx(img_grid_skeletion, horizontal, cv::MORPH_OPEN, horizontal_kernel, cv::Point(-1, -1), 1);

    // 수직선 감지
    cv::Mat vertical;
    cv::Mat vertical_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 5));
    cv::morphologyEx(img_grid_skeletion, vertical, cv::MORPH_OPEN, vertical_kernel, cv::Point(-1, -1), 1);


    // 교차점 감지
    cv::Mat joints;
    cv::bitwise_and(horizontal, vertical, joints);

    
    // 합치기
    cv::Mat new_line;
    cv::bitwise_or(horizontal, vertical, new_line);
    cv::imshow("new_line", new_line);


    cv::imshow("Joints", joints);  

    cv::imshow("Horizontal", horizontal);
    cv::imshow("Vertical", vertical);
    
    
    std::vector<cv::Point> featurePts;
    cv::Mat img_horizontal= cv::Mat::zeros(img_wall_skeletion.size(), CV_8UC1);

    std::vector<cv::Point> horizontalPts = makeImagetoPoint(horizontal);
    vector<vector<Point>> segHorizontalPts;  
    findConnectedComponents(horizontalPts,  segHorizontalPts); 

    cout << "segHorizontalPts.size(): " <<segHorizontalPts.size() << endl;
    for (size_t i =0; i<segHorizontalPts.size(); i++)
    {
        cout<<"num_segHorizontalPts " << segHorizontalPts[i].size() << endl;

        for (const auto &pt : segHorizontalPts[i])       
        {
            cv::circle(img_horizontal, pt, 0, cv::Scalar(0, 255, 0), -1);

            int y = pt.y;
            int x = pt.x;
            img_horizontal.at<uchar>(y, x) = 255;

        } 

        int index_end = segHorizontalPts[i].size()-1;
        cv::Point start = segHorizontalPts[i][0];
        cv::Point end = segHorizontalPts[i][index_end];

        
        featurePts.push_back(start);
        featurePts.push_back(end);


        cv::circle(test_rotated_img, start, 3, cv::Scalar(0, 255, 0), -1);
        cv::circle(test_rotated_img, end, 3, cv::Scalar(0, 0, 255), -1);
    }


    std::vector<cv::Point> verticalPts = makeImagetoPoint(vertical);
    vector<vector<Point>> segVerticalPts;  
    findConnectedComponents(verticalPts,  segVerticalPts); 

    
    cout << "segVerticalPts.size(): " <<segVerticalPts.size() << endl;
    for (size_t i =0; i<segVerticalPts.size(); i++)
    {
        cout<<"num_segVerticalPts " << segVerticalPts[i].size() << endl;

        // for (const auto &pt : segVerticalPts[i])       
        // {
        //     cv::circle(img_horizontal, pt, 0, cv::Scalar(0, 255, 0), -1);

        //     int y = pt.y;
        //     int x = pt.x;
        //     img_horizontal.at<uchar>(y, x) = 255;

        // } 

        int index_end = segVerticalPts[i].size()-1;
        cv::Point start = segVerticalPts[i][0];
        cv::Point end = segVerticalPts[i][index_end];

        cv::circle(color_rotated_raw_img, start, 3, cv::Scalar(255, 255, 0), -1);
        cv::circle(color_rotated_raw_img, end, 3, cv::Scalar(255, 0, 255), -1);

        featurePts.push_back(start);
        featurePts.push_back(end);

        cv::circle(test_rotated_img, start, 3, cv::Scalar(255, 0, 0), -1); 
        cv::circle(test_rotated_img, end, 3, cv::Scalar(0, 0, 255), -1); 

    }
 


    removeDuplicatePoints(featurePts);
    std::vector<cv::Point> sorted_featurePts = sortPoints(featurePts);



    std::cout << std::endl;

    cv::Mat img_freeSpace = cv::Mat::zeros(rotated_raw_img.size(), CV_8UC1);    
    for (int i =0; i<img_freeSpace.rows; i++)
    {
        for (int j=0; j<img_freeSpace.cols; j++)
        {
            if (rotated_raw_img.at<uchar>(i, j) >= 220)
            {
                img_freeSpace.at<uchar>(i, j) = 255;
            }
        }
    }
    cv::imshow("img_freeSpace", img_freeSpace);


     // 확실한 전경 영역 찾기
    std::cout <<"makeDistanceTransform()..." << std::endl;

    cv::Mat dist_transform;
    cv::distanceTransform(img_freeSpace, dist_transform, cv::DIST_L2, 3);    
    normalize(dist_transform, dist_transform, 0, 255, cv::NORM_MINMAX);
    
    cv::Mat dist_transform_8u;    
    dist_transform.convertTo(dist_transform_8u, CV_8UC1);
    cv::imshow("distTransform", dist_transform_8u);
 

    cv::Mat img_freeSpace_skeletion;
    zhangSuenThinning(dist_transform_8u, img_freeSpace_skeletion);
    cv::imshow("img_freeSpace_skeletion", img_freeSpace_skeletion);


    std::vector<cv::Point> trajectoryPts;
    for (int i = 0; i < img_freeSpace_skeletion.rows; i++)
    {
        for (int j = 0; j < img_freeSpace_skeletion.cols; j++)
        {

            if (img_freeSpace_skeletion.at<uchar>(i, j) == 255)
            {
                trajectoryPts.push_back(cv::Point(j, i));
            }
        }
    }
    std::vector<cv::Point> sorted_trajectoryPts = sortPoints(trajectoryPts);
    
    for (const auto &pt : sorted_trajectoryPts)
    {
        cv::circle(color_rotated_raw_img, pt, 1, CV_RGB(255, 0, 0), -1);
    }

    int radius = 20; // 탐색 범위 반지름
    vector<Point> circlesCenters = addHalfOverlappingCircles(sorted_trajectoryPts, radius);

    
    //std::vector<SEGDATA> database;
    std::vector<std::vector<LINEINFO>> lineDBdata;

    for (size_t i = 0; i < circlesCenters.size();)
    {
        cv::Point exploreCenterPt = circlesCenters[i];
        
        cv::Mat img_update = test_rotated_img.clone();         
        cv::drawMarker(img_update, exploreCenterPt, cv::Scalar(0, 0, 255), cv::MARKER_CROSS);

        SEGDATA db = testExploreFeature3(sorted_featurePts, sorted_trajectoryPts, exploreCenterPt, radius);

        // 지정한 윈도우 안에 feture point 2개 이하 이며 탐색 제외
        if (db.feturePoints.size() < 2)
        {
            circlesCenters.erase(circlesCenters.begin() + i);
        }
        else
        {
        
            // drawingSetpCircule(result_img, cp, radius);
            drawingSetpRectangle(img_update, exploreCenterPt, radius);

            //db.addState(true);
            std::cout << "-----------------------------------------------------" << std::endl;
            std::cout << "Center Point:(" << db.centerPoint.x << ", " << db.centerPoint.y << ")\n";

            std::cout << "TrajectoryPts:";  
            for (const auto &pt : db.trajectoryPoints)
            {

                cv::circle(img_update, pt, 1, cv::Scalar(255, 0, 0), -1);
                std::cout << "(" << pt.x << ", " << pt.y << ") ";
            };
            std::cout << std::endl;
            std::cout << "FeaturePt:\n";
            for (const auto &pt : db.feturePoints)
            {
                cv::circle(img_update, pt, 3, cv::Scalar(0, 255, 0), -1);
                std::cout << "(" << pt.x << ", " << pt.y << ") ";
            }

            // std::cout << std::endl;
            // mergeClosePoints(db.feturePoints, 3);
            // for (const auto &pt : db.feturePoints)
            // {
            //     std::cout << "(" << pt.x << ", " << pt.y << ") ";
            // }
            // std::cout << std::endl;
            //---------------------------------------------------------------------------------------------------------
           
            // 거리 계산 함수 호출
            // calculateDistancesWithBIntersect(db.feturePoints, db.trajectoryPoints);

            std::vector<LINEINFO> seglines = checkPointsNearLineSegments(db.feturePoints, db.trajectoryPoints, 3);            
            lineDBdata.push_back(seglines);

            std::cout << std::endl;
            std::cout << "-----------------------------------------------------" << std::endl;
            ++i;
            
            //database.push_back(db);
        }
        // cv::imshow("img_update", img_update);
        // cv::waitKey(0);
    }

    std::cout << "------------------------------------------------------------------------" << std::endl;
    std::cout << "lineDBdata.size(): " << lineDBdata.size() <<std::endl;
        
    for (size_t i=0; i<lineDBdata.size(); i++)
    {
        for (size_t j = 0; j<lineDBdata[i].size(); j++)
        {
            cv::Point startPt = lineDBdata[i][j].virtual_wll.first;
            cv::Point endPt = lineDBdata[i][j].virtual_wll.second;
            double distance = lineDBdata[i][j].distance;

            std::cout << "Line: (" << startPt.x << ", " << startPt.y << ") - (" 
                      << endPt.x << ", " << endPt.y
                      << ") distance: " << distance << std::endl;
        }  
    }
    std::cout <<"==============================================================" <<std::endl;
    

    std::vector<LINEINFO> db = convertToLineInfo(lineDBdata);
    std::cout <<"==============================================================" <<std::endl;
    // Remove duplicates
    std::vector<LINEINFO> uniqueLines = removeDuplicates(db);

    // Output the unique values
    for (const auto& line : uniqueLines) {
        std::cout << "Line: ((" << line.virtual_wll.first.x << ", " << line.virtual_wll.first.y << "), ("
                  << line.virtual_wll.second.x << ", " << line.virtual_wll.second.y << ")) - Distance: "
                  << line.distance << std::endl;
    }

    std::cout <<"==============================================================" <<std::endl;
    // Filtering lines based on the criteria
    std::vector<LINEINFO> filteredLines = filterLines(uniqueLines);
        
    // Output the filtered lines
    for (const auto& line : filteredLines) {
        std::cout << "Line: ("
                  << line.virtual_wll.first.x << ", " << line.virtual_wll.first.y << ") to ("
                  << line.virtual_wll.second.x << ", " << line.virtual_wll.second.y << ") - Distance: "
                  << line.distance << std::endl;

        if (line.distance < 20.0)
        {
            cv::line(img_freeSpace, line.virtual_wll.first, line.virtual_wll.second, cv::Scalar(0), 1);
            cv::line(test_rotated_img, line.virtual_wll.first, line.virtual_wll.second, CV_RGB(0, 255, 0), 2);
        }
        
    }


    cv::imshow("test_rotated_img", test_rotated_img);  
    cv::imshow("img_freeSpace", img_freeSpace);    

    std::cout << img_freeSpace.channels() << endl;


    

    // 5. 원본 이미지에 외곽선을 그립니다.
    cv::Mat result_image;
    cv::cvtColor(img_freeSpace, result_image, cv::COLOR_GRAY2BGR);  // 색상을 추가하여 결과를 시각화

    cv::Mat color_img_grid;
    cv::cvtColor(img_grid, color_img_grid, cv::COLOR_GRAY2BGR);    


    // 3. 레이블링 작업을 수행합니다 (8방향 연결).
    cv::Mat labels, stats, centroids;
    int n_labels = cv::connectedComponentsWithStats(img_freeSpace, labels, stats, centroids, 4, CV_32S);
    std::cout << "n_labels: " << n_labels << std::endl;


    std::vector<cv::Rect> rectangles;
    for (int label = 1; label < n_labels; ++label) 
    {
        // 바운딩 박스 좌표
        int x = stats.at<int>(label, cv::CC_STAT_LEFT);
        int y = stats.at<int>(label, cv::CC_STAT_TOP);
        int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);   

        Rect ret(x, y, width, height);
        // 바운딩 박스 그리기        
        rectangles.push_back(ret);            
        cv::rectangle(color_img_grid, ret, Scalar(0, 255, 0), 3); 
        cv::rectangle(img_grid, ret, Scalar(255), 3); 
    }  

    cv::imshow("color_img_grid", color_img_grid);
    cv::imshow("img_grid2", img_grid);

    cv::Mat img_grid_skeletion2;
    zhangSuenThinning(img_grid, img_grid_skeletion2);
    cv::imshow("img_grid_skeletion2", img_grid_skeletion2);



for (const auto &rect: rectangles) 
{
    if (rect.x >= 0 && rect.y >= 0 && rect.x + rect.width <= test_rotated_img2.cols && rect.y + rect.height <= test_rotated_img2.rows) {
        for (int y = rect.y; y < rect.y + rect.height; ++y) {
            for (int x = rect.x; x < rect.x + rect.width; ++x) {
                
                if (img_grid.at<uchar>(y, x) == 0) {
                    test_rotated_img2.at<cv::Vec3b>(y, x)= cv::Vec3b(0, 0, 255);                    
                    // cv::Scalar(0, 0, 255); //randomColor();  // 새로운 색상으로 변경
                }
            }
        } 
    }
}

cv::imshow("test_rotated_img2", test_rotated_img2);






/*

 // 결과를 저장할 벡터 (rectangles 개수만큼)
    std::vector<std::vector<cv::Point>> classifiedPoints(rectangles.size());

    // 각 feature point를 Rects에 대해 검사
    for (const auto& pt : featurePts) {
        for (size_t i = 0; i < rectangles.size(); ++i) {
            if (rectangles[i].contains(pt)) { // 점이 Rect 안에 있는지 확인
                classifiedPoints[i].push_back(pt); // 포함된 점 저장
                //break; // 한 Rect에 포함되면 더 이상 체크할 필요 없음
            }
        }
    }



    // 결과 출력
    for (size_t i = 0; i < rectangles.size(); ++i) {

        cv::Mat update_img = result_image.clone();
        std::cout << "Rect " << i + 1 << "에 포함된 점들:" << std::endl;
        
        cv::rectangle(update_img, rectangles[i], cv::Scalar(0, 255, 0), 2);
        Scalar color = randomColor();
        for (const auto& pt : classifiedPoints[i]) {            
            cv::circle(update_img, pt, 3,color, -1);            
            std::cout << pt << std::endl;            
        }
        cv::imshow("update_img", update_img);
        waitKey(0);
    }

*/






    // std::vector<std::vector<cv::Point>> roomContours;
    // // 5. 각 레이블에 대해 개별적으로 윤곽선 검출
    // for (int label = 1; label < n_labels; ++label) {  // 0번 레이블은 배경이므로 제외

    //     // 5.1. 현재 레이블에 해당하는 영역을 추출
    //     cv::Mat label_mask = (labels == label);  // 현재 레이블에 해당하는 부분만 추출 (이진화)

    //     // 5.2. 윤곽선 검출
    //     std::vector<std::vector<cv::Point>> contours;
    //     std::vector<cv::Vec4i> hierarchy;
    //     cv::findContours(label_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    //     // 윤곽선 저장
    //     roomContours.insert(roomContours.end(), contours.begin(), contours.end());

    //     // 5.3. 각 레이블에 대한 윤곽선을 빨간색으로 그립니다.
    //     cv::drawContours(result_image, contours, -1, randomColor(), 2);
        
    //     // // std::vector<cv::Point> roomtemp;
    //     // // for (int i =0; i< contours.size(); i++)
    //     // // {
    //     // //     for (int j=0; j<contours[i].size(); j++)
    //     // //     {
    //     // //         //std::cout<< "[" << label << "] [ " <<i <<" ]" << contours[i][j] << std::endl;
    //     // //        // roomtemp.push_back(contours[i][j]);
    //     // //     }
    //     // // }
    //     // // roomContours.push_back(roomtemp); 
    // }
    // // 6. 윤곽선 좌표 출력
    // for (size_t i = 0; i < roomContours.size(); ++i) {
    //     std::cout << "윤곽선 " << i + 1 << "의 좌표들:" << std::endl;
    //     for (size_t j = 0; j < roomContours[i].size(); ++j) {
    //         std::cout << "(" << roomContours[i][j].x << ", " << roomContours[i][j].y << ") ";
    //     }
    //     std::cout << std::endl << std::endl;
    // }

    // std::cout << "Feature Pts" << std::endl;
    // for (const auto &pt : sorted_featurePts)
    // {
    //     std::cout << "( " << pt.x << "," << pt.y << ") ";
    // }





    
    cv::imshow("result_image", result_image);    
    
    

    cv::waitKey(0);
    return 0;
}
