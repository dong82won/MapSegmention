#include "featuredetection.h"

FeatureDetection::FeatureDetection(const cv::Mat &img, std::vector<cv::Point> &featurePoints) 
                                   :img_(img), featurePoints_(featurePoints)
//FeatureDetection::FeatureDetection(const cv::Mat &img):img_(img)
{
}

FeatureDetection::~FeatureDetection()
{
}

void FeatureDetection::imgShow(std::vector<cv::Point> &updateFeaturePoint)
{
    cv::Mat img_color;
    cv::cvtColor(img_, img_color, cv::COLOR_GRAY2BGR);

    for (const auto &pt :  updateFeaturePoint)
    {
        cv::circle(img_color, pt, 3, CV_RGB(255, 0, 0), -1);
    }
    
    cv::imshow("result_featurePoints", img_color);
    cv::waitKey(0);
}

cv::Mat FeatureDetection::straightLineDetection()
{

    cv::Mat src_gray;
    img_.convertTo(src_gray, CV_64FC1);

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
        cv::line(lsd, pt1, pt2, cv::Scalar(255), 1, cv::LINE_8);
    }
    free_ntuple_list(ntl);

    return lsd;
}

//--------------------------------------------------------------------------------
// 거리 내의 점들을 병합하는 함수
void FeatureDetection::mergeClosePoints(std::vector<cv::Point> &points, int distanceThreshold)
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
//--------------------------------------------------------------------------------
// End-points를 감지하고 그 좌표를 vector에 저장
void FeatureDetection::detectEndPoints(const cv::Mat &imgLine, int distanceThreshold)
{
    for (int y = 1; y < imgLine.rows - 1; ++y)
    {
        for (int x = 1; x < imgLine.cols - 1; ++x)
        {
            if (imgLine.at<uchar>(y, x) == 255)
            {
                int count = 0;

                // 8방향 이웃 확인
                count += imgLine.at<uchar>(y - 1, x - 1) == 255 ? 1 : 0;
                count += imgLine.at<uchar>(y - 1, x) == 255 ? 1 : 0;
                count += imgLine.at<uchar>(y - 1, x + 1) == 255 ? 1 : 0;
                count += imgLine.at<uchar>(y, x + 1) == 255 ? 1 : 0;
                count += imgLine.at<uchar>(y + 1, x + 1) == 255 ? 1 : 0;
                count += imgLine.at<uchar>(y + 1, x) == 255 ? 1 : 0;
                count += imgLine.at<uchar>(y + 1, x - 1) == 255 ? 1 : 0;
                count += imgLine.at<uchar>(y, x - 1) == 255 ? 1 : 0;

                // End-point는 이웃이 하나만 있는 픽셀
                if (count == 1)
                {
                    featurePoints_.push_back(cv::Point(x, y));
                }
            }
        }
    }

    // 교차점 병합
    mergeClosePoints(featurePoints_, 12); // 3 픽셀 이내의 점을 병합
}

//--------------------------------------------------------------------------------
// 최소값을 가진 픽셀의 위치를 계산하는 함수
cv::Point FeatureDetection::calculateMinimumPointAround(cv::Point pt)
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
            if (nx >= 0 && ny >= 0 && nx < img_.cols && ny < img_.rows)
            {
                double currentValue = static_cast<double>(img_.at<uchar>(ny, nx));
                if (currentValue < minValue)
                {
                    minValue = currentValue;
                    minPoint = cv::Point(nx, ny);

                    sum += img_.at<uchar>(ny, nx); // 픽셀 값 누적
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


//--------------------------------------------------------------------------------
// 최소값을 가진 픽셀의 위치를 계산하는 함수
std::vector<cv::Point> FeatureDetection::updateFeaturePoints()
{
    std::vector<cv::Point> result_points;

    for (const auto &pt : featurePoints_)
    {
        // cv::circle(color_raw_img1, pt, 3, cv::Scalar(0, 0, 255), -1);
        int pixelValue = img_.at<uchar>(pt.y, pt.x);        

        cv::Point min_Point = calculateMinimumPointAround(pt);
        
        int min_pixelValue = img_.at<uchar>(min_Point.y, min_Point.x);
        
        // std::cout << "org. : " << pt << ", mP: " << min_Point << std::endl;
        // std::cout << "pixelValue: " << pixelValue << ", min_pixelValue: " << min_pixelValue << std::endl;

        if (min_pixelValue < 20)
        {
            //cv::circle(result_img, pt, 3, CV_RGB(255, 0, 0), -1);
            result_points.push_back(pt);
        }
    }
    return result_points;
}
