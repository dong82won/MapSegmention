#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <iostream>
#include <stack>
#include <cmath>



// #include <random>
// #include <set>
// #include <utility> // For std::pair
// #include <map>
// #include <set>
// #include "FAST.hpp"

#include "lsd.h"

using namespace cv;
using namespace std;

// Custom comparator for cv::Point
struct ComparePoints {
    bool operator()(const Point &p1, const Point &p2) const {
        if (p1.x != p2.x)
            return p1.x < p2.x;
        return p1.y < p2.y;
    }
};

class Graph {
public:
    // adjacencyList는 각 정점과 그 정점에 연결된 이웃 정점들을 저장하는 인접 리스트
    map<Point, set<Point, ComparePoints>, ComparePoints> adjacencyList;

    // 그래프에 정점을 추가합니다.
    void addVertex(const Point &p) {
        adjacencyList[p]; // 정점을 추가 (이미 존재하면 무시됨)
    }

    // 두 정점을 연결하는 간선을 추가합니다.
    void addEdge(const Point &p1, const Point &p2) {
        adjacencyList[p1].insert(p2);
        adjacencyList[p2].insert(p1); // 무방향 그래프의 경우
    }

    // 그래프를 출력합니다.
    void printGraph() const {
        for (const auto &node : adjacencyList) {
            cout << "Vertex " << node.first << " is connected to: ";
            for (const auto &neighbor : node.second) {
                cout << neighbor << " ";
            }
            cout << endl;
        }
    }

};

/**/
void buildGraphFromImage(Graph &graph, const Mat &edgesImage, const vector<Point> &points) {
    vector<vector<Point>> contours;
    findContours(edgesImage, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    cout << "contours.size(): " << contours.size() << endl;


    // 그래프에 정점 추가
    for (const auto &point : points) {
        graph.addVertex(point);
    }

    // 직선이 어떤 점들을 연결하는지 분석하여 그래프에 간선 추가
    for (const auto &contour : contours) {
        for (size_t i = 0; i < contour.size(); ++i) {
            Point pt1 = contour[i];
            Point pt2 = contour[(i + 1) % contour.size()]; // 다음 점, 마지막 점은 첫 점과 연결

            Point p1Closest, p2Closest;
            double minDist1 = DBL_MAX, minDist2 = DBL_MAX;

            // 각 점에 대해 선분의 양 끝 점과 가장 가까운 점을 찾음
            for (const auto &pt : points) {
                double dist1 = norm(pt - pt1);
                double dist2 = norm(pt - pt2);

                if (dist1 < minDist1) {
                    minDist1 = dist1;
                    p1Closest = pt;
                }
                if (dist2 < minDist2) {
                    minDist2 = dist2;
                    p2Closest = pt;
                }
            }

            // 일정 거리 내에 있는 경우 두 점을 그래프에 간선으로 추가
            //if (minDist1 < 10 && minDist2 < 10) { // 10은 임계값, 필요에 따라 조정
                graph.addEdge(p1Closest, p2Closest);
            //}
        }
    }
}

double euclideanDistance(const cv::Point &p1, const cv::Point &p2)
{
    return cv::norm(p1 - p2);
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
 
    

/* */
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


/**/
// 8방향 이웃의 개수를 계산하여 끝점 또는 교차점을 찾기 위한 함수
int countNonZeroNeighbors(const Mat &img, int row, int col)
{
    int count = 0;
    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            if (i == 0 && j == 0)
                continue;
            if (img.at<uchar>(row + i, col + j) > 0)
            {
                count++;
            }
        }
    }
    return count;
}

vector<Point> extractBendingAndEndPoints(cv::Mat Skeleton_dst)
{
    // 꺾이는 지점과 끝점을 저장할 벡터
    vector<Point> trajector_points; 

    // 각 픽셀을 순회하면서 꺾이는 지점 찾기
    
    for (int row = 1; row < Skeleton_dst.rows - 1; row++)
    {
        for (int col = 1; col < Skeleton_dst.cols - 1; col++)
        {
            if (Skeleton_dst.at<uchar>(row, col) > 0)
            {
                int neighborCount = countNonZeroNeighbors(Skeleton_dst, row, col);
                if (neighborCount == 1 || neighborCount > 2)
                //if ( neighborCount > 2)
                { // 끝점(1개 이웃) 또는 교차점(3개 이상의 이웃)
                    trajector_points.push_back(Point(col, row));
                }
            }
        }
    }

    return trajector_points;
}

cv::Mat MakeDistanceTransform(cv::Mat freeSpace)
{
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 3));    
    cv::Mat opening_freeSpace;
    cv::morphologyEx(freeSpace, opening_freeSpace, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);
    //cv::imshow("opening_freeSpace", opening_freeSpace);


    // 확실한 전경 영역 찾기
    cv::Mat dist_transform;
    cv::distanceTransform(opening_freeSpace, dist_transform, cv::DIST_L2, 3);
    normalize(dist_transform, dist_transform, 0, 255, NORM_MINMAX);

    cv::Mat dist_transform_8u;
    // 전경 이미지를 8비트로 변환
    dist_transform.convertTo(dist_transform_8u, CV_8UC1);
    cv::imshow("distTransform", dist_transform_8u);

    return dist_transform_8u;
}



bool isAdjacent(const cv::Point &pt1, const cv::Point &pt2)
{
    // 두 점이 8방향으로 인접해 있는지 확인
    return std::abs(pt1.x - pt2.x) <= 1 && std::abs(pt1.y - pt2.y) <= 1;
}

/* */
// DFS로 연속된 선 성분 찾기
void findConnectedComponent(const std::vector<cv::Point> &coordinates,
                            std::vector<bool> &visited,
                            std::vector<cv::Point> &component,
                            size_t startIdx)
{
    std::stack<size_t> stack;
    stack.push(startIdx);

    while (!stack.empty())
    {
        size_t currentIdx = stack.top();
        stack.pop();

        if (visited[currentIdx])
            continue;

        visited[currentIdx] = true;
        component.push_back(coordinates[currentIdx]);

        // 현재 픽셀과 인접한 픽셀을 찾는다
        for (size_t i = 0; i < coordinates.size(); ++i)
        {
            if (!visited[i] && isAdjacent(coordinates[currentIdx], coordinates[i]))
            {
                stack.push(i);
            }
        }
    }
}


/**/
std::vector<cv::Point> processTrajectoryFeaturePoints(const std::vector<cv::Point> &coordinates)
{
    std::vector<cv::Point> result;
    std::vector<bool> visited(coordinates.size(), false);

    for (size_t i = 0; i < coordinates.size(); ++i)
    {
        if (!visited[i])
        {
            std::vector<cv::Point> component;
            findConnectedComponent(coordinates, visited, component, i);

            if (!component.empty())
            {
                // 각 선 성분의 시작과 끝 점을 결과에 추가
                result.push_back(component.front());
                result.push_back(component.back());
            }
        }
    }

    mergeClosePoints(result, 12);

    return result;
}


int main()
{

    // 이미지 파일 경로
    cv::Mat raw_img = cv::imread("/home/dongwon/myWorkCode/regonSeg/imgdb/occupancy_grid.png", cv::IMREAD_GRAYSCALE);
    //cv::Mat raw_img = cv::imread("/home/dongwon/myWorkCode/regonSeg/imgdb/caffe_map.pgm", cv::IMREAD_GRAYSCALE);
    if (raw_img.empty())
    {
        std::cerr << "Error: Unable to open image file: " << std::endl;
        return -1;
    }
    cv::imshow("raw_img", raw_img);

    cv::Mat color_raw_img;
    cv::cvtColor(raw_img, color_raw_img, cv::COLOR_GRAY2RGB);
    
    //----------------------------------------------------------------------------------------
    //----------------------------------------------------------------------------------------

    cv::Mat back_img = cv::imread("/home/dongwon/myWorkCode/regonSeg/imgdb/occupancy_grid_back.png", cv::IMREAD_GRAYSCALE);
    cv::imshow("back_img", back_img);
    if (back_img.empty())
    {
        std::cerr << "Error: 이미지를 열 수 없습니다!" << std::endl;
        return -1;
    }

    cv::Mat dist_transform = MakeDistanceTransform(back_img);        
    //----------------------------------------------------------------------------------------
    cv::Mat Skeleton_dst;
    zhangSuenThinning(dist_transform, Skeleton_dst); 
    cv::imshow("pass_Skeleton", Skeleton_dst);    
    //----------------------------------------------------------------------------------------
    // 꺾이는 지점과 끝점을 저장할 벡터
    vector<Point> trajector_points = extractBendingAndEndPoints(Skeleton_dst);     
    // 좌표 데이터 처리
    std::vector<cv::Point> trajectoryfeatures = processTrajectoryFeaturePoints(trajector_points);    
    // 그래프 객체 생성
    Graph graph;
    // 이미지를 바탕으로 그래프 구축
    buildGraphFromImage(graph, Skeleton_dst, trajectoryfeatures);    

    // 그래프 출력
    // graph.printGraph();   

    // 연결된 점들 간의 선을 이미지에 그리기
    std::vector<cv::Point> vertex;
    for (const auto &node : graph.adjacencyList)
    {        
        if (node.second.size() > 0)
        {
            cout << "Vertex " << node.first << " is connected to: ";
            cout << "num: " << node.second.size() << endl;

            for (const auto &neighbor : node.second)
            {
                if (node.first != neighbor)
                {
                    line(color_raw_img, node.first, neighbor, Scalar(255, 0, 255), 2); // 빨간색 선으로 표시                    
                    cout << neighbor << " ";
                    
                    double dist = euclideanDistance(node.first, neighbor);
                    cout << dist << " ";

                    circle(color_raw_img, node.first, 3, CV_RGB(0, 255, 0), -1); 
                    
                    vertex.push_back(node.first);
                }
            }
        }
        cout << endl;
    }

    imshow("color_raw_img", color_raw_img);   
    waitKey(0); 


    //주요
    //cout <<"----------------------------------------------------------------" << endl;
    /*
    cv::Mat color_raw_img2 = color_raw_img1.clone();
    cout <<"vertex.size(): " << vertex.size() << endl;

    
    map<Point, vector<Point>, PointCompare2> result;
    int rangeX = 20;
    int rangeY = 20;

    //findAndStorePoints(vertex, skeleton_point, result, rangeX, rangeY);

    vector<cv::Point> p1 = vertex;
    cout <<"p1.size(): " << p1.size() << endl;

    vector<cv::Point> p2 = skeleton_point;
    
     for (const Point& pt1 : p1) {
        vector<Point> candidates;

        // p1 점의 주변에서 p2 점을 찾음
        for (const Point& pt2 : p2) {
            // 범위 내에 있는지 확인 (p1 점을 중심으로 하는 사각형)
            if (pt2.x >= pt1.x - rangeX && pt2.x <= pt1.x + rangeX &&
                pt2.y >= pt1.y - rangeY && pt2.y <= pt1.y + rangeY) {
                
                cv::rectangle(color_raw_img1, Point(pt1.x - rangeX, pt1.y - rangeY), Point(pt1.x + rangeX, pt1.y + rangeY), CV_RGB(255, 0, 0));

                candidates.push_back(pt2);
            }
        }

        if (candidates.size() > 1) {
            // 거리 기반으로 정렬
            sort(candidates.begin(), candidates.end(), [&](const Point& a, const Point& b) {
                return calculateDistance(pt1, a) < calculateDistance(pt1, b);
            });

            // 가장 가까운 점 두 개를 찾기
            vector<Point> closestPoints;
            if (candidates.size() >= 2) {
                closestPoints.push_back(candidates[0]);
                closestPoints.push_back(candidates[1]);
            }

            // 결과 저장
            if (closestPoints.size() > 1) {
                result[pt1] = closestPoints;
            }
        }
    }    

    // 결과 출력
    for (const auto& pair : result) {
        cout << "p1 Point: (" << pair.first.x << ", " << pair.first.y << ")" << endl;

        cv::rectangle(color_raw_img2, Point(pair.first.x - rangeX, pair.first.y - rangeY), Point(pair.first.x + rangeX, pair.first.y + rangeY), CV_RGB(255, 0, 0));

        for (const Point& pt : pair.second) {
            cout << "  Matching p2 Point: (" << pt.x << ", " << pt.y << ")" << endl;

        }
    }
    cout <<"----------------------------------------------------------------" << endl;
    imshow("color_raw_img2", color_raw_img2);    
    //주요
    */
     
     
    return 0;
}
