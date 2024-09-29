#include <opencv2/opencv.hpp>

#include <vector>
#include <iostream>
#include <random>
#include <cmath>

//#include "utility.h"
#include "roomSeg.h"


using namespace cv;
using namespace std;


cv::Mat extractWallElements(const cv::Mat &occupancyMap, uchar thread_wall_value = 64)
{
    int rows = occupancyMap.rows;
    int cols = occupancyMap.cols;

    cv::Mat wall_img = cv::Mat::zeros(occupancyMap.size(), CV_8UC1);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            uchar pixelValue = occupancyMap.at<uchar>(i, j);

            if (pixelValue < thread_wall_value)
            {
                wall_img.at<uchar>(i, j) = 255;
            }
        }
    }

    return wall_img;
}

// 연결된 성분을 찾는 함수
void findConnectedComponents(const vector<Point> &contour, vector<vector<Point>> &components)
{

    // 체인 코드 방향 (8방향)
    int dx[] = {1, 1, 0, -1, -1, -1, 0, 1};
    int dy[] = {0, 1, 1, 1, 0, -1, -1, -1};

    vector<bool> visited(contour.size(), false);

    // 현재 성분을 저장할 변수
    vector<Point> currentComponent;

    for (size_t i = 0; i < contour.size(); i++)
    {
        if (visited[i])
            continue; // 이미 방문한 점은 건너뜀

        // 새로운 성분 시작
        currentComponent.clear();
        vector<Point> stack; // DFS를 위한 스택
        stack.push_back(contour[i]);

        while (!stack.empty())
        {
            Point p = stack.back();
            stack.pop_back();

            // 현재 점을 성분에 추가하고 방문 처리
            currentComponent.push_back(p);
            int idx = find(contour.begin(), contour.end(), p) - contour.begin();
            visited[idx] = true;

            // 8방향으로 탐색
            for (int dir = 0; dir < 8; dir++)
            {
                Point neighbor(p.x + dx[dir], p.y + dy[dir]);

                // 인접 점이 윤곽선에 존재하고 방문하지 않았을 경우 스택에 추가
                if (find(contour.begin(), contour.end(), neighbor) != contour.end() && !visited[find(contour.begin(), contour.end(), neighbor) - contour.begin()])
                {
                    stack.push_back(neighbor);
                }
            }
        }

        // 연결된 성분을 저장
        components.push_back(currentComponent);
    }
}

// 격자 점 계산 함수
Point calculateSnappedPoint(const Point &pixel, int gridSize)
{
    int snappedX = round(static_cast<double>(pixel.x) / gridSize) * gridSize;
    int snappedY = round(static_cast<double>(pixel.y) / gridSize) * gridSize;
    return Point(snappedX, snappedY);
}

void gridSnapping(const Mat &inputImage, Mat &outputImage, int gridSize)
{

    // 결과 이미지를 0으로 초기화
    outputImage = Mat::zeros(inputImage.size(), inputImage.type());

    // 이미지에서 모든 픽셀을 순회
    for (int y = 0; y < inputImage.rows; y++)
    {
        for (int x = 0; x < inputImage.cols; x++)
        {

            // 흰색 픽셀(255)인 경우
            if (inputImage.at<uchar>(y, x) == 255)
            {

                // 가장 가까운 격자 점 계산
                Point snappedPoint = calculateSnappedPoint(Point(x, y), gridSize);

                // 격자 점으로 픽셀 이동
                int halfGridSize = (gridSize / 2) + 1;

                // 지정된 격자 크기만큼의 영역을 흰색으로 설정
                for (int dy = -halfGridSize; dy < halfGridSize; dy++)
                {
                    for (int dx = -halfGridSize; dx < halfGridSize; dx++)
                    {
                        int newY = snappedPoint.y + dy;
                        int newX = snappedPoint.x + dx;

                        // 이미지 경계 내에서만 설정
                        if (newX >= 0 && newX < outputImage.cols &&
                            newY >= 0 && newY < outputImage.rows)
                        {
                            outputImage.at<uchar>(newY, newX) = 255; // 흰색 픽셀로 설정
                        }
                    }
                }
            }
        }
    }
}


// 두 원형 영역이 반만 겹치는지 확인하는 함수
bool isHalfOverlap(const Point &center1, int radius1, const Point &center2, int radius2)
{
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


// 두 원형 영역이 겹치는지 확인하는 함수
bool isOverlap(const Point& center1, int radius1, const Point& center2, int radius2) {
    double distance = sqrt(pow(center1.x - center2.x, 2) + pow(center1.y - center2.y, 2));
    return distance < (radius1 + radius2);
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


// x, y 범위 내에 포함되는 포인트를 찾는 함수
std::vector<cv::Point> findPointsInRange(const std::vector<cv::Point> &points, int x_min, int x_max, int y_min, int y_max)
{

    std::vector<cv::Point> filteredPoints;

    // std::copy_if를 사용하여 조건에 맞는 점들만 필터링
    std::copy_if(points.begin(), points.end(), std::back_inserter(filteredPoints),
                 [x_min, x_max, y_min, y_max](const cv::Point &pt)
                 {
                     return (pt.x >= x_min && pt.x <= x_max && pt.y >= y_min && pt.y <= y_max);
                 });

    return filteredPoints;
}

SEGDATA exploreFeaturePoint(std::vector<cv::Point> &feature_points,
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
                      << end.x << ", " << end.y << ") = distance " << calculateDistance(start, end) << "\n";

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
            else
            {
                std::cout << "    No points from dataB are near this line.\n";
            }
        }
    }

    return lines;
}

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

// Function to check if two LINEINFO objects are equal regardless of the order of points
bool areEqualIgnoringOrder(const LINEINFO &a, const LINEINFO &b)
{
    return (a.virtual_wll == b.virtual_wll) ||
           (a.virtual_wll.first == b.virtual_wll.second && a.virtual_wll.second == b.virtual_wll.first);
}

// Function to remove duplicates from a vector of LINEINFO
std::vector<LINEINFO> removeLineDuplicates(const std::vector<LINEINFO> &lines)
{
    std::set<LINEINFO, LineInfoComparator> uniqueLines;

    for (const auto &line : lines)
    {
        bool isDuplicate = false;
        for (const auto &uniqueLine : uniqueLines)
        {
            if (areEqualIgnoringOrder(line, uniqueLine))
            {
                isDuplicate = true;
                break;
            }
        }
        if (!isDuplicate)
        {
            uniqueLines.insert(line);
        }
    }

    // Convert back to vector
    return std::vector<LINEINFO>(uniqueLines.begin(), uniqueLines.end());
}

// std::vector<std::vector<LINEINFO>>를 std::vector<LINEINFO>로 변환하는 함수
std::vector<LINEINFO> convertToLineInfo(const std::vector<std::vector<LINEINFO>> &a)
{
    std::vector<LINEINFO> b; // 1D 벡터를 위한 벡터
    // 2D 벡터를 순회하며 각 요소를 1D 벡터에 추가
    for (const auto &innerVector : a)
    {
        b.insert(b.end(), innerVector.begin(), innerVector.end());
    }

    return b; // 변환된 벡터 반환
}


// 중복되는 점과 점은 거리가 큰 것은 제거함
//------------------------------------------------------------------------------
// Check if two lines overlap based on their endpoints
bool linesOverlap(const LINEINFO& line1, const LINEINFO& line2) {
    return (line1.virtual_wll.first == line2.virtual_wll.first || 
            line1.virtual_wll.first == line2.virtual_wll.second ||
            line1.virtual_wll.second == line2.virtual_wll.first || 
            line1.virtual_wll.second == line2.virtual_wll.second);
}
 
// Filter lines based on overlapping conditions
std::vector<LINEINFO> finalFillterLine(std::vector<LINEINFO>& lines) {
    
    std::vector<LINEINFO> result;

    for (size_t i = 0; i < lines.size(); ++i) {
        bool toRemove = false;
        
        for (size_t j = 0; j < lines.size(); ++j) {
            if (i != j && linesOverlap(lines[i], lines[j])) {
                if (lines[i].distance > lines[j].distance) {
                    toRemove = true; // Mark for removal
                    break;
                }
            }
        }
        
        if (!toRemove) {
            result.push_back(lines[i]); // Keep line if it is not marked for removal
        }
    }
    
    return result;
}


int main()
{

    //1. 이미지 입력 및 회전 -------------------------------------------------------
    std::string home_path = getenv("HOME");
    // std::cout << home_path << std::endl;

    // 이미지 파일 경로
    cv::Mat raw_img = cv::imread(home_path + "/myStudyCode/MapSegmention/imgdb/occupancy_grid.png", cv::IMREAD_GRAYSCALE);    
    //cv::Mat raw_img = cv::imread(home_path + "/myStudyCode/MapSegmention/imgdb/caffe_map.pgm", cv::IMREAD_GRAYSCALE);

    if (raw_img.empty())
    {
        std::cerr << "Error: Unable to open image file: " << std::endl;
        return -1;
    }
    imshow("raw_img", raw_img);

    cv::Mat img_wall = extractWallElements(raw_img); //(occupancyMap);
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
    //이미지 입력 및 회전 -------------------------------------------------------
    


    //2. 윤곽선 단수화 ---------------------------------------------------------
    cv::Mat color_rotated_raw_img;
    cv::cvtColor(rotated_raw_img, color_rotated_raw_img, COLOR_GRAY2BGR);

    cv::Mat test_rotated_img = color_rotated_raw_img.clone();
    cv::Mat test_rotated_img2 = color_rotated_raw_img.clone();

    cv::Mat img_wall_skeletion;
    zhangSuenThinning(rotatedImage, img_wall_skeletion);
    cv::imshow("img_wall_skeletion", img_wall_skeletion);

    std::vector<cv::Point> edgePts = changeMatoPoint(img_wall_skeletion);

    cv::Mat img_test = cv::Mat::zeros(img_wall_skeletion.size(), CV_8UC3);
    cv::Mat img_edge = cv::Mat::zeros(img_wall_skeletion.size(), CV_8UC1);

    // 연결된 성분을 저장할 벡터
    vector<vector<Point>> segEdgePts;
    findConnectedComponents(edgePts, segEdgePts);

    cout << "segEdgePts.size(): " << segEdgePts.size() << endl;
    for (size_t i = 0; i < segEdgePts.size(); i++)
    {
        cout << "edge_num: " << segEdgePts[i].size() << endl;
        if (segEdgePts[i].size() > 20)
        {
            for (const auto &pt : segEdgePts[i])
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
    //2.------------------------------------------------------------------------
    


    //3. Feature Poits 추출------------------------------------------------------
    std::vector<cv::Point> featurePts;
    cv::Mat img_horizontal = cv::Mat::zeros(img_wall_skeletion.size(), CV_8UC1);

    std::vector<cv::Point> horizontalPts = changeMatoPoint(horizontal);
    vector<vector<Point>> segHorizontalPts;
    findConnectedComponents(horizontalPts, segHorizontalPts);

    cout << "segHorizontalPts.size(): " << segHorizontalPts.size() << endl;
    for (size_t i = 0; i < segHorizontalPts.size(); i++)
    {
        cout << "num_segHorizontalPts " << segHorizontalPts[i].size() << endl;

        for (const auto &pt : segHorizontalPts[i])
        {
            cv::circle(img_horizontal, pt, 0, cv::Scalar(0, 255, 0), -1);

            int y = pt.y;
            int x = pt.x;
            img_horizontal.at<uchar>(y, x) = 255;
        }

        int index_end = segHorizontalPts[i].size() - 1;
        cv::Point start = segHorizontalPts[i][0];
        cv::Point end = segHorizontalPts[i][index_end];

        featurePts.push_back(start);
        featurePts.push_back(end);

        cv::circle(test_rotated_img, start, 3, cv::Scalar(0, 255, 0), -1);
        cv::circle(test_rotated_img, end, 3, cv::Scalar(0, 0, 255), -1);
    }

    std::vector<cv::Point> verticalPts = changeMatoPoint(vertical);
    vector<vector<Point>> segVerticalPts;
    findConnectedComponents(verticalPts, segVerticalPts);

    cout << "segVerticalPts.size(): " << segVerticalPts.size() << endl;
    for (size_t i = 0; i < segVerticalPts.size(); i++)
    {
        cout << "num_segVerticalPts " << segVerticalPts[i].size() << endl; 

        int index_end = segVerticalPts[i].size() - 1;
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
    //3. Feature Poits 추출------------------------------------------------------

    //4. 주행 궤적 정보  ----------------------------------------------------------
    cv::Mat img_freeSpace = cv::Mat::zeros(rotated_raw_img.size(), CV_8UC1);
    for (int i = 0; i < img_freeSpace.rows; i++)
    {
        for (int j = 0; j < img_freeSpace.cols; j++)
        {
            if (rotated_raw_img.at<uchar>(i, j) >= 130)
            {
                img_freeSpace.at<uchar>(i, j) = 255;
            }
        }
    }
    cv::imshow("img_freeSpace", img_freeSpace);
    // 확실한 전경 영역 찾기
    std::cout << "makeDistanceTransform()..." << std::endl;

    cv::Mat dist_transform;
    cv::distanceTransform(img_freeSpace, dist_transform, cv::DIST_L2, 3);
    normalize(dist_transform, dist_transform, 0, 255, cv::NORM_MINMAX);
    
    cv::Mat dist_transform_8u;
    dist_transform.convertTo(dist_transform_8u, CV_8UC1);
    cv::imshow("distTransform", dist_transform_8u);
    
    cv::Mat img_dist_bin;
    threshold(dist_transform_8u, img_dist_bin, 0, 255, THRESH_BINARY | THRESH_OTSU);
    cv::imshow("img_dist_bin", img_dist_bin);

    cv::Mat img_freeSpace_skeletion;
    zhangSuenThinning(img_dist_bin, img_freeSpace_skeletion);
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
    //4. 주행 가능 정보  ----------------------------------------------------------



    //5. Exploring partition 탐색 분할 영역 ---------------------------------------
    int radius = 20; // 탐색 범위 반지름
    vector<Point> circlesCenters = addHalfOverlappingCircles(sorted_trajectoryPts, radius);

    // std::vector<SEGDATA> database;
    std::vector<std::vector<LINEINFO>> lineDBdata;

    for (size_t i = 0; i < circlesCenters.size();)
    {
        cv::Point exploreCenterPt = circlesCenters[i];

        cv::Mat img_update = test_rotated_img.clone();
        cv::drawMarker(img_update, exploreCenterPt, cv::Scalar(0, 0, 255), cv::MARKER_CROSS);

        SEGDATA db = exploreFeaturePoint(sorted_featurePts, sorted_trajectoryPts, exploreCenterPt, radius);

        // 지정한 윈도우 안에 feture point 2개 이하 이며 탐색 제외
        if (db.feturePoints.size() < 2)
        {
            circlesCenters.erase(circlesCenters.begin() + i);
        }
        else
        {

            // drawingSetpCircule(result_img, cp, radius);
            drawingSetpRectangle(img_update, exploreCenterPt, radius);

            // db.addState(true);
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

            // database.push_back(db);
        }
        cv::imshow("img_update", img_update);
        cv::waitKey(0);
    }
    //5. Exploring partition 탐색 분할 영역 ---------------------------------------


    //6. 문턱 가능한 데이터 필터링 --------------------------------------------------
    std::cout << "Loding...Line DB data.." << std::endl;
    std::cout << "lineDBdata.size(): " << lineDBdata.size() << std::endl;

    for (size_t i = 0; i < lineDBdata.size(); i++)
    {
        std::cout << "lineDBdata [ " << i << "]"<<  std::endl;
        for (size_t j = 0; j < lineDBdata[i].size(); j++)
        {
            cv::Point startPt = lineDBdata[i][j].virtual_wll.first;
            cv::Point endPt = lineDBdata[i][j].virtual_wll.second;
            double distance = lineDBdata[i][j].distance;

            std::cout << "Line: (" << startPt.x << ", " << startPt.y << ") - ("
                      << endPt.x << ", " << endPt.y
                      << ") distance: " << distance << std::endl;
        }
    }    
    std::vector<LINEINFO> db = convertToLineInfo(lineDBdata); 


    std::cout << "removeLineDuplicates()....===========================================" << std::endl;
    // Remove duplicates
    std::vector<LINEINFO> uniqueLines = removeLineDuplicates(db);

    // Output the unique values
    for (const auto &line : uniqueLines)
    {
        std::cout << "Line: ((" << line.virtual_wll.first.x << ", " << line.virtual_wll.first.y << "), ("
                  << line.virtual_wll.second.x << ", " << line.virtual_wll.second.y << ")) - Distance: "
                  << line.distance << std::endl;
    }

    std::cout << "==============================================================" << std::endl;
    std::vector<LINEINFO> filteredLines = finalFillterLine(uniqueLines);

    // Output the filtered lines
    for (const auto &line : filteredLines)
    {
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
    //6. 문턱 가능한 데이터 필터링 --------------------------------------------------



    // std::cout << img_freeSpace.channels() << endl;

    // // 5. 원본 이미지에 외곽선을 그립니다.
    // cv::Mat result_image;
    // cv::cvtColor(img_freeSpace, result_image, cv::COLOR_GRAY2BGR); // 색상을 추가하여 결과를 시각화

    // cv::Mat color_img_grid;
    // cv::cvtColor(img_grid, color_img_grid, cv::COLOR_GRAY2BGR);

    // // 3. 레이블링 작업을 수행합니다 (8방향 연결).
    // cv::Mat labels, stats, centroids;
    // int n_labels = cv::connectedComponentsWithStats(img_freeSpace, labels, stats, centroids, 4, CV_32S);
    // std::cout << "n_labels: " << n_labels << std::endl;

    // std::vector<cv::Rect> rectangles;
    // for (int label = 1; label < n_labels; ++label)
    // {
    //     // 바운딩 박스 좌표
    //     int x = stats.at<int>(label, cv::CC_STAT_LEFT);
    //     int y = stats.at<int>(label, cv::CC_STAT_TOP);
    //     int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
    //     int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);

    //     Rect ret(x, y, width, height);
    //     // 바운딩 박스 그리기
    //     rectangles.push_back(ret);
    //     cv::rectangle(color_img_grid, ret, Scalar(0, 255, 0), 3);
    //     cv::rectangle(img_grid, ret, Scalar(255), 3);
    // }

    // cv::imshow("color_img_grid", color_img_grid);
    // cv::imshow("img_grid2", img_grid);

    // cv::Mat img_grid_skeletion2;
    // zhangSuenThinning(img_grid, img_grid_skeletion2);
    // cv::imshow("img_grid_skeletion2", img_grid_skeletion2);

    // for (const auto &rect : rectangles)
    // {
    //     if (rect.x >= 0 && rect.y >= 0 && rect.x + rect.width <= test_rotated_img2.cols && rect.y + rect.height <= test_rotated_img2.rows)
    //     {
    //         for (int y = rect.y; y < rect.y + rect.height; ++y)
    //         {
    //             for (int x = rect.x; x < rect.x + rect.width; ++x)
    //             {

    //                 if (img_grid.at<uchar>(y, x) == 0)
    //                 {
    //                     test_rotated_img2.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
    //                     // cv::Scalar(0, 0, 255); //randomColor();  // 새로운 색상으로 변경
    //                 }
    //             }
    //         }
    //     }
    // // }

    // cv::imshow("test_rotated_img2", test_rotated_img2);
    // cv::imshow("result_image", result_image);
    cv::waitKey(0);

    return 0;
}
