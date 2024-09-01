#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <iostream>
#include <stack>
#include <cmath>


#include "trajectioryPoint.h"

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

int main()
{

    std::string home_path = getenv("HOME");
    std::cout << home_path << std::endl;

    // 이미지 파일 경로
    cv::Mat raw_img = cv::imread(home_path + "/myStudyCode/regonSeg/imgdb/occupancy_grid.png", cv::IMREAD_GRAYSCALE);
    //cv::Mat raw_img = cv::imread(home_path + "/myStudyCode/regonSeg/imgdb/caffe_map.pgm", cv::IMREAD_GRAYSCALE);
    if (raw_img.empty())
    {
        std::cerr << "Error: Unable to open image file: " << std::endl;
        return -1;
    }

    cv::Mat img_freeSpace = makeFreeSpace(raw_img);
    cv::imshow("img_freeSpace", img_freeSpace);
     
    cv::Mat color_raw_img;
    cv::cvtColor(raw_img, color_raw_img, cv::COLOR_GRAY2RGB);    

    // cv::Mat img_freeSpace = cv::imread(home_path + "/myStudyCode/regonSeg/imgdb/occupancy_grid_back.png", cv::IMREAD_GRAYSCALE);
    // cv::imshow("back_img", img_freeSpace);
    // if (img_freeSpace.empty())
    // {
    //     std::cerr << "Error: 이미지를 열 수 없습니다!" << std::endl;
    //     return -1;
    // }


    TrajectionPoint tp;
    cv::Mat img_dist= tp.makeDistanceTransform(img_freeSpace);

    cv::Mat img_skeletion;
    tp.zhangSuenThinning(img_dist, img_skeletion); 
    cv::imshow("img_skeletion", img_skeletion);      

    // 꺾이는 지점과 끝점을 저장할 벡터
    std::vector<cv::Point> trajector_points = tp.extractBendingAndEndPoints(img_skeletion);     
    
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
                    line(color_raw_img, node.first, neighbor, cv::Scalar(255, 0, 255), 2);
                    std::cout << neighbor << " ";
                    
                    double dist = tp.euclideanDistance(node.first, neighbor);
                    std::cout << dist << " ";

                    circle(color_raw_img, node.first, 3, CV_RGB(0, 255, 0), -1); 
                    
                    vertex.push_back(node.first);
                }
            }
        }
        std::cout << std::endl;
    }

    cv::imshow("color_raw_img", color_raw_img);    
    cv::waitKey();  

 

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
