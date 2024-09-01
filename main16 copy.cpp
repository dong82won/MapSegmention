#include <opencv2/opencv.hpp>

#include <vector>
#include <iostream>
#include <random>
#include <cmath>

#include "lsd.h"

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


cv::Mat straightLineDetection(cv::Mat &src)
{

    cv::Mat src_gray;
    src.convertTo(src_gray, CV_64FC1);

    int cols = src_gray.cols;
    int rows = src_gray.rows;

    image_double image = new_image_double(cols, rows);
    image->data = src_gray.ptr<double>(0);

    ntuple_list ntl = lsd(image);

    cv::Mat lsd = cv::Mat::zeros(rows, cols, CV_8UC1);
    cv::Point pt1, pt2;
    for (int j = 0; j != ntl->size; ++j)
    {
        pt1.x = ntl->values[0 + j * ntl->dim];
        pt1.y = ntl->values[1 + j * ntl->dim];
        pt2.x = ntl->values[2 + j * ntl->dim];
        pt2.y = ntl->values[3 + j * ntl->dim];

        double width = ntl->values[4 + j * ntl->dim];
        cv::line(lsd, pt1, pt2, cv::Scalar(255), 1, LINE_8);
    }
    free_ntuple_list(ntl);

    return lsd;
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

// 최소값을 가진 픽셀의 위치를 계산하는 함수
cv::Point calculateMinimumPointAround(const cv::Mat &image, cv::Point pt)
{

    int halfSize = 5; // windowSize / 2;
    
    double sum = 0.0;
    int count = 0;

    double minValue = std::numeric_limits<double>::max();
    cv::Point minPoint(-1, -1);

    for (int i = -halfSize; i <= halfSize; i++)
    {
        for (int j = -halfSize; j <= halfSize; j++)
        {
            int nx = pt.x + i;
            int ny = pt.y + j;

            // 이미지 경계 체크
            if (nx >= 0 && ny >= 0 && nx < image.cols && ny < image.rows)
            {
                double currentValue = static_cast<double>(image.at<uchar>(ny, nx));
                if (currentValue < minValue)
                {
                    minValue = currentValue;
                    minPoint = cv::Point(nx, ny);

                    sum += image.at<uchar>(ny, nx); // 픽셀 값 누적
                    count++;
                }
            }
        }
    }

    // 결과가 유효한 경우 반환
    if (minPoint.x == -1 || minPoint.y == -1)
    {
        throw std::runtime_error("No valid minimum point found.");
    }
    return minPoint;
}

int main()
{

    // 이미지 파일 경로
    std::string home_path = getenv("HOME");

    cv::Mat raw_img = cv::imread(home_path + "/myWorkCode/regonSeg/imgdb/occupancy_grid.png", cv::IMREAD_GRAYSCALE);
    // cv::Mat raw_img = cv::imread(home_path +"/myWorkCode/regonSeg/imgdb/caffe_map.pgm", cv::IMREAD_GRAYSCALE);
    if (raw_img.empty())
    {
        std::cerr << "Error: Unable to open image file: " << std::endl;
        return -1;
    }
    cv::imshow("raw_img", raw_img);

    cv::Mat result_img;
    cv::cvtColor(raw_img, result_img, cv::COLOR_GRAY2RGB);

    //----------------------------------------------------------------------------------------
    cv::Mat lsd_img = straightLineDetection(raw_img);
    cv::imshow("lsd_img", lsd_img);

    std::vector<cv::Point> skeleton_point;
    detectEndPoints(lsd_img, skeleton_point);

    vector<cv::Point> result_points;
    for (const auto &pt : skeleton_point)
    {
        // cv::circle(color_raw_img1, pt, 3, cv::Scalar(0, 0, 255), -1);
        int pixelValue = raw_img.at<uchar>(pt.y, pt.x);        
        cv::Point min_Point = calculateMinimumPointAround(raw_img, pt);
        
        int min_pixelValue = raw_img.at<uchar>(min_Point.y, min_Point.x);
        cout << "org. : " << pt << ", mP: " << min_Point << endl;
        cout << "pixelValue: " << pixelValue << ", min_pixelValue: " << min_pixelValue << endl;

        if (min_pixelValue < 20)
        {
            cv::circle(result_img, pt, 3, CV_RGB(255, 0, 0), -1);
            result_points.push_back(pt);
        }
    }

    imshow("result_img", result_img);
    waitKey(0);

    return 0;
}