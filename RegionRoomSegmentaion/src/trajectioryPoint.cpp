#include "trajectioryPoint.h"

TrajectionPoint::TrajectionPoint() 
{

}
TrajectionPoint::~TrajectionPoint() 
{

}


cv::Mat TrajectionPoint::makeDistanceTransform(cv::Mat freeSpace)
{
    // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));    
    // cv::Mat opening_freeSpace;
    // cv::morphologyEx(freeSpace, opening_freeSpace, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);
    //cv::imshow("opening_freeSpace", opening_freeSpace);

    // 확실한 전경 영역 찾기
    std::cout <<"makeDistanceTransform()..." << std::endl;

    cv::Mat dist_transform;
    cv::distanceTransform(freeSpace, dist_transform, cv::DIST_L2, 3);
    normalize(dist_transform, dist_transform, 0, 255, cv::NORM_MINMAX);

    cv::Mat dist_transform_8u;
    // 전경 이미지를 8비트로 변환
    dist_transform.convertTo(dist_transform_8u, CV_8UC1);
    cv::imshow("distTransform", dist_transform_8u);

    return dist_transform_8u;
}


// Zhang-Suen Thinning Algorithm
void TrajectionPoint::zhangSuenThinning(const cv::Mat &src, cv::Mat &dst)
{

    cv::Mat img;
    int th = (int)cv::threshold(src, img, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::imshow("img", img);

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


/**/
// 8방향 이웃의 개수를 계산하여 끝점 또는 교차점을 찾기 위한 함수
int TrajectionPoint::countNonZeroNeighbors(const cv::Mat &img, int row, int col)
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

std::vector<cv::Point> TrajectionPoint::extractBendingAndEndPoints(cv::Mat src, std::vector<cv::Point> &trajector_lines)
{
    // 꺾이는 지점과 끝점을 저장할 벡터
    std::vector<cv::Point> trajector_points;
    //std::vector<cv::Point> trajector_lines;  

    // 각 픽셀을 순회하면서 꺾이는 지점 찾기    
    for (int row = 1; row < src.rows - 1; row++)
    {
        for (int col = 1; col < src.cols - 1; col++)
        {
            if (src.at<uchar>(row, col) > 0)
            {
                int neighborCount = countNonZeroNeighbors(src, row, col);
                
                //if (neighborCount > 2)
                if (neighborCount == 1 || neighborCount > 2)
                { // 끝점(1개 이웃) 또는 교차점(3개 이상의 이웃)
                    trajector_points.push_back(cv::Point(col, row));
                }
                if (src.at<uchar>(row, col) == 255)
                {
                    trajector_lines.push_back(cv::Point(col, row));
                }
            }
        }
    }
    return trajector_points;
}


bool TrajectionPoint::isAdjacent(const cv::Point &pt1, const cv::Point &pt2)
{
    // 두 점이 8방향으로 인접해 있는지 확인
    return std::abs(pt1.x - pt2.x) <= 1 && std::abs(pt1.y - pt2.y) <= 1;
}

/* */
// DFS로 연속된 선 성분 찾기
void TrajectionPoint::findConnectedComponent(const std::vector<cv::Point> &coordinates,
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


/* */
// 거리 내의 점들을 병합하는 함수
void TrajectionPoint::mergeClosePoints(std::vector<cv::Point> &points, int distanceThreshold)
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
std::vector<cv::Point> TrajectionPoint::processTrajectoryFeaturePoints(const std::vector<cv::Point> &coordinates)
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

    mergeClosePoints(result, 9);
    
    return result;
}


/**/
void TrajectionPoint::buildGraphFromImage(Graph &graph, const cv::Mat &edgesImage, const std::vector<cv::Point> &points) 
{
    std::vector<std::vector<cv::Point>> contours;
    findContours(edgesImage, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    std::cout << "contours.size(): " << contours.size() << std::endl;

    // 그래프에 정점 추가
    for (const auto &point : points) {
        graph.addVertex(point);
    }

    // 직선이 어떤 점들을 연결하는지 분석하여 그래프에 간선 추가
    for (const auto &contour : contours) {
        for (size_t i = 0; i < contour.size(); ++i) {
            cv::Point pt1 = contour[i];
            cv::Point pt2 = contour[(i + 1) % contour.size()]; // 다음 점, 마지막 점은 첫 점과 연결

            cv::Point p1Closest, p2Closest;
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
            //if (minDist1 > 6 && minDist2 > 6) { // 10은 임계값, 필요에 따라 조정
                graph.addEdge(p1Closest, p2Closest);
            //}
        }
    }
}

double TrajectionPoint::euclideanDistance(const cv::Point &p1, const cv::Point &p2)
{
    return cv::norm(p1 - p2);
}






