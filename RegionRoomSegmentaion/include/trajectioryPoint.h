#ifndef TRAJECTIONRYPOINT_H
#define TRAJECTIONRYPOINT_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <iostream>
#include <stack>
#include <cmath>


// Custom comparator for cv::Point
struct ComparePoints {
    bool operator()(const cv::Point &p1, const cv::Point &p2) const {
        if (p1.x != p2.x)
            return p1.x < p2.x;
        return p1.y < p2.y;
    }
};

class Graph {
public:
    // adjacencyList는 각 정점과 그 정점에 연결된 이웃 정점들을 저장하는 인접 리스트
    std::map<cv::Point, std::set<cv::Point, ComparePoints>, ComparePoints> adjacencyList;

    // 그래프에 정점을 추가합니다.
    void addVertex(const cv::Point &p) {
        adjacencyList[p]; // 정점을 추가 (이미 존재하면 무시됨)
    }

    // 두 정점을 연결하는 간선을 추가합니다.
    void addEdge(const cv::Point &p1, const cv::Point &p2) {
        adjacencyList[p1].insert(p2);
        adjacencyList[p2].insert(p1); // 무방향 그래프의 경우
    }

    // 그래프를 출력합니다.
    void printGraph() const {
        for (const auto &node : adjacencyList) {
            std::cout << "Vertex " << node.first << " is connected to: ";
            for (const auto &neighbor : node.second) {
                std::cout << neighbor << " ";
            }
            std::cout << std::endl;
        }
    }

};

class TrajectionPoint
{
    private:
    
        int countNonZeroNeighbors(const cv::Mat &img, int row, int col);
        
        bool isAdjacent(const cv::Point &pt1, const cv::Point &pt2);
        void findConnectedComponent(const std::vector<cv::Point> &coordinates,
                                std::vector<bool> &visited,
                                std::vector<cv::Point> &component,
                                size_t startIdx);
        void mergeClosePoints(std::vector<cv::Point> &points, int distanceThreshold);    
    
    
    public:
        TrajectionPoint();
        ~TrajectionPoint();

        cv::Mat makeDistanceTransform(cv::Mat freeSpace);
        void zhangSuenThinning(const cv::Mat &src, cv::Mat &dst);
        std::vector<cv::Point> extractBendingAndEndPoints(cv::Mat src, std::vector<cv::Point> &component);    
        void buildGraphFromImage(Graph &graph, const cv::Mat &edgesImage, const std::vector<cv::Point> &points) ;
        double euclideanDistance(const cv::Point &p1, const cv::Point &p2);
        std::vector<cv::Point> processTrajectoryFeaturePoints(const std::vector<cv::Point> &coordinates);
};




#endif