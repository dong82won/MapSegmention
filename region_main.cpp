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
            //if (pixelValue > 128) {
            if (pixelValue > 205) {
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


int main()
{


    std::string home_path = getenv("HOME");
    //std::cout << home_path << std::endl;
    
    // 이미지 파일 경로
    cv::Mat raw_img = cv::imread(home_path + "/myStudyCode/MapSegmention/imgdb/occupancy_grid.png", cv::IMREAD_GRAYSCALE);
    //cv::Mat raw_img = cv::imread(home_path + "/myStudyCode/regonSeg/imgdb/caffe_map.pgm", cv::IMREAD_GRAYSCALE);
    if (raw_img.empty())
    {
        std::cerr << "Error: Unable to open image file: " << std::endl;
        return -1;
    }
    cv::Mat result_img;
    cv::cvtColor(raw_img, result_img, cv::COLOR_GRAY2RGB);

    std::vector<cv::Point> featurePoints;

    FeatureDetection fd(raw_img, featurePoints);
    cv::Mat img_lsd = fd.straightLineDetection();

    fd.detectEndPoints(img_lsd, 12);
    std::vector<cv::Point> updata_featurePoints = fd.updateFeaturePoints();

    //fd.imgShow(updata_featurePoints);
 
    //--------------------------------------------------------------------------    
    cv::Mat img_freeSpace = makeFreeSpace(raw_img);
    cv::imshow("img_freeSpace", img_freeSpace);


    TrajectionPoint tp;
    cv::Mat img_dist= tp.makeDistanceTransform(img_freeSpace);

    cv::Mat img_skeletion;
    tp.zhangSuenThinning(img_dist, img_skeletion); 
    cv::imshow("img_skeletion", img_skeletion);      

    // 꺾이는 지점과 끝점을 저장할 벡터
    std::vector<cv::Point> trajector_line;
    std::vector<cv::Point> trajector_points = tp.extractBendingAndEndPoints(img_skeletion, trajector_line);         

    cv::Mat test(img_skeletion.size(), CV_8UC3, CV_RGB(0, 0, 0));
    
    for (const auto &pt : trajector_line)
    {
        cv::circle(test, pt, 1, cv::Scalar(255, 255, 255), -1 );
    }
    for (const auto &pt : trajector_points)
    {
        cv::circle(test, pt, 3, cv::Scalar(0, 255, 0), -1 );
    }
    cv::imshow("test", test);      


    // 좌표 데이터 처리
    std::vector<cv::Point> trajectory_features = tp.processTrajectoryFeaturePoints(trajector_points);    

    // 그래프 객체 생성
    Graph graph;
    // 이미지를 바탕으로 그래프 구축
    tp.buildGraphFromImage(graph, img_skeletion, trajectory_features);    

    // 연결된 점들 간의 선을 이미지에 그리기
    std::vector<cv::Point> vertex;
    for (const auto &node : graph.adjacencyList)
    {        
        if (node.second.size() > 0)
        {
            std::cout << "Vertex " << node.first << " is connected to: ";
            std::cout << "num: " << node.second.size() << std::endl;

            for (const auto &neighbor : node.second)
            {
                if (node.first != neighbor)
                {
                    line(result_img, node.first, neighbor, cv::Scalar(255, 0, 255), 2);
                    std::cout << neighbor << " ";
                    
                    double dist = tp.euclideanDistance(node.first, neighbor);
                    std::cout << dist << " ";

                    circle(result_img, node.first, 3, CV_RGB(0, 255, 0), -1); 
                    
                    vertex.push_back(node.first);
                }
            }
        }
        std::cout << std::endl;
    }

    for (const auto &fp : updata_featurePoints)
    {
        circle(result_img, fp, 3, CV_RGB(255, 0, 0), -1); 
    }

    cv::imshow("result_img", result_img);    
    
    //--------------------------------------------------------------------------

    //주요
    //cout <<"----------------------------------------------------------------" << endl;
   
   cv::Mat color_img;
    cv::cvtColor(raw_img, color_img, cv::COLOR_GRAY2RGB);
    
    map<Point, vector<Point>, PointCompare> result;

    int rangeX = 15;
    int rangeY = 15;

    vector<cv::Point> p1 = vertex;
    cout <<"p1.size(): " << p1.size() << endl;
    vector<cv::Point> p2 = updata_featurePoints;
    cout <<"p2.size(): " << p2.size() << endl;
    
     for (const Point& pt1 : p1) {
        vector<Point> candidates;

        // p1 점의 주변에서 p2 점을 찾음
        for (const Point& pt2 : p2) {
            // 범위 내에 있는지 확인 (p1 점을 중심으로 하는 사각형)
            if (pt2.x >= pt1.x - rangeX && pt2.x <= pt1.x + rangeX &&
                pt2.y >= pt1.y - rangeY && pt2.y <= pt1.y + rangeY) {
                
                //cv::rectangle(color_img, Point(pt1.x - rangeX, pt1.y - rangeY), Point(pt1.x + rangeX, pt1.y + rangeY), CV_RGB(255, 0, 0));

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
    for (const auto& pt1 : result) {
        cout << "p1 Point: (" << pt1.first.x << ", " << pt1.first.y << ")" << endl;
        cv::rectangle(color_img, cv::Point(pt1.first.x - rangeX, pt1.first.y - rangeY), 
                                 cv::Point(pt1.first.x + rangeX, pt1.first.y + rangeY), CV_RGB(255, 0, 0));
        
        circle(color_img, pt1.first, 3, CV_RGB(255, 0, 0), -1); 

        for (const Point& pt2 : pt1.second) {
            cout << "  Matching p2 Point: (" << pt2.x << ", " << pt2.y << ")" << endl;
            cv::circle(color_img, pt2, 3, CV_RGB(0, 255, 255), -1); 
            cv::line(color_img, pt1.first, pt2, CV_RGB(0, 255, 0));
        }
    }
    cout <<"----------------------------------------------------------------" << endl;
    imshow("color_img", color_img);    


    cv::waitKey();  

    return 0;
}



    //주요
    
