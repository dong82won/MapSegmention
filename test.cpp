#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <set>

using namespace cv;
using namespace std;

// 유클리드 거리 계산 함수
double distance(const Point& p1, const Point& p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

// 커스텀 비교 함수: 두 cv::Point를 비교
struct PointComparator {
    bool operator()(const Point& p1, const Point& p2) const {
        return tie(p1.x, p1.y) < tie(p2.x, p2.y); // 먼저 x좌표 비교, 같으면 y좌표 비교
    }
};

// 가까운 교차점을 병합하는 함수
vector<Point> mergeClosePoints(const vector<Point>& points, double minDist) {
    vector<Point> mergedPoints;
    vector<bool> visited(points.size(), false);

    for (size_t i = 0; i < points.size(); i++) {
        if (visited[i]) continue;
        
        Point sum = points[i];
        int count = 1;
        
        for (size_t j = i + 1; j < points.size(); j++) {
            if (distance(points[i], points[j]) < minDist) {
                sum += points[j]; // 가까운 교차점들을 병합
                visited[j] = true;
                count++;
            }
        }

        mergedPoints.push_back(sum / count); // 평균 좌표로 병합된 교차점 저장
    }
    
    return mergedPoints;
}

// 끝점과 연결된 교차점을 찾는 함수 (한 쌍으로 저장)
vector<pair<Point, Point>> findConnectedIntersections(const map<Point, set<Point, PointComparator>, PointComparator>& adjacencyMap, const vector<Point>& endpoints, const vector<Point>& intersections) {
    vector<pair<Point, Point>> endpointIntersectionPairs;

    // 끝점에서 출발하는 경로를 탐색
    for (const Point& endpoint : endpoints) {
        set<Point, PointComparator> visited; // 방문한 점들을 추적하기 위한 집합
        vector<Point> stack = {endpoint}; // DFS를 위한 스택
        bool foundIntersection = false;

        // DFS를 통해 끝점에서 교차점까지 경로 탐색
        while (!stack.empty() && !foundIntersection) {
            Point current = stack.back();
            stack.pop_back();

            // 교차점에 도달하면 해당 교차점을 저장하고 탐색 종료
            if (find(intersections.begin(), intersections.end(), current) != intersections.end()) {
                endpointIntersectionPairs.push_back(make_pair(endpoint, current));
                foundIntersection = true;
                break;
            }

            // 현재 점과 연결된 점들을 스택에 추가
            for (const Point& neighbor : adjacencyMap.at(current)) {
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    stack.push_back(neighbor);
                }
            }
        }
    }

    return endpointIntersectionPairs;
}

int main() {
    // 번개 패턴의 실선 좌표 데이터 (예시)
    vector<Point> lightningPattern = {
        Point(10, 10), Point(50, 100), Point(100, 50),
        Point(150, 150), Point(200, 80), Point(250, 200),
        Point(150, 150), Point(100, 200), Point(50, 250),
        Point(50, 100), Point(100, 200)
    };

    // 인접성 그래프를 구성하기 위한 map (cv::Point를 키로 사용하며 커스텀 비교 함수 제공)
    map<Point, set<Point, PointComparator>, PointComparator> adjacencyMap;

    // 인접성 관계를 설정 (각 점에서 서로 연결된 점 추가)
    for (size_t i = 0; i < lightningPattern.size() - 1; i++) {
        adjacencyMap[lightningPattern[i]].insert(lightningPattern[i + 1]);
        adjacencyMap[lightningPattern[i + 1]].insert(lightningPattern[i]);
    }

    vector<Point> endpoints;
    vector<Point> intersections;

    // 끝점과 교차점 계산
    for (const auto& pair : adjacencyMap) {
        const Point& point = pair.first;
        const set<Point, PointComparator>& connections = pair.second;

        if (connections.size() == 1) {
            // 연결된 선분이 1개인 경우, 끝점으로 처리
            endpoints.push_back(point);
        } else if (connections.size() > 2) {
            // 연결된 선분이 3개 이상인 경우, 교차점으로 처리
            intersections.push_back(point);
        }
    }

    // 교차점 병합 (거리가 minDist 이하인 경우 병합)
    double minDist = 15.0;
    intersections = mergeClosePoints(intersections, minDist);

    // 끝점과 연결된 교차점 찾기
    vector<pair<Point, Point>> endpointIntersectionPairs = findConnectedIntersections(adjacencyMap, endpoints, intersections);

    // 결과 출력 (끝점과 연결된 교차점 쌍)
    cout << "Endpoint and Connected Intersections: " << endl;
    for (const auto& pair : endpointIntersectionPairs) {
        cout << "Endpoint: (" << pair.first.x << ", " << pair.first.y << ")"
             << " -> Intersection: (" << pair.second.x << ", " << pair.second.y << ")" << endl;
    }

    // 시각화를 위한 이미지 생성
    Mat img(300, 300, CV_8UC3, Scalar(0, 0, 0));//
    
    // 실선 그리기
    for (size_t i = 0; i < lightningPattern.size(); i++) {
       // line(img, lightningPattern[i], lightningPattern[i + 1], Scalar(255, 255, 255), 2);

        circle(img, lightningPattern[i], 5, Scalar(0, 255, 0), -1);
    }

    // // 끝점 표시 (녹색 원)
    // for (const auto& pt : endpoints) {
    //     circle(img, pt, 5, Scalar(0, 255, 0), -1);
    // }

    // // 병합된 교차점 표시 (빨간 원)
    // for (const auto& pt : intersections) {
    //     circle(img, pt, 5, Scalar(0, 0, 255), -1);
    // }

    // // 끝점과 연결된 교차점 표시 (파란 선으로 표시)
    // for (const auto& pair : endpointIntersectionPairs) {
    //     line(img, pair.first, pair.second, Scalar(255, 0, 0), 2);
    // }

    // 결과 이미지 표시
    imshow("Lightning Pattern", img);
    waitKey(0);
    return 0;
}
