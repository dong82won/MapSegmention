#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{

    std::string home_path = getenv("HOME");
    // 이미지 파일 경로
    cv::Mat src = cv::imread(home_path + "/myStudyCode/MapSegmention/imgdb/coins.jpeg");

    if (src.empty())
    {
        std::cerr << "Error: Unable to load image!" << std::endl;
        return -1;
    }

    // 원본 이미지 보여주기
    cv::imshow("Original Image", src);

    // 1. 그레이스케일 이미지 변환
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);


    // 2. 이진화 (Thresholding)
    cv::Mat thresh;
    cv::threshold(gray, thresh, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU); // Otsu 이진화

    // 3. 노이즈 제거를 위해 모폴로지 열림 연산 적용
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    
    cv::Mat opening;
    cv::morphologyEx(thresh, opening, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 2); // 열림 연산 적용

    // 4. 확실한 배경 영역 찾기 (팽창)
    cv::Mat sure_bg;
    cv::dilate(opening, sure_bg, kernel, cv::Point(-1, -1), 3); // 배경 확장


    // 5. 확실한 전경 영역 찾기 (거리 변환 후 임계값 적용)
    cv::Mat dist_transform;
    cv::distanceTransform(opening, dist_transform, cv::DIST_L2, 5); // 거리 변환

    cv::Mat sure_fg;
    cv::threshold(dist_transform, sure_fg, 0.7 * cv::norm(dist_transform, cv::NORM_INF), 255, 0); // 전경 영역 찾기
    sure_fg.convertTo(sure_fg, CV_8U);                                                            // 8비트로 변환


    cv::imshow("sure_bg", sure_bg);
    cv::imshow("sure_fg", sure_fg);

    // 6. 불확실한 영역 찾기 (배경 - 전경)
    cv::Mat unknown = sure_bg - sure_fg;

    cv::imshow("unknown", unknown);



    // 7. 전경 영역 레이블링 (Connected Components)
    cv::Mat markers;
    cv::connectedComponents(sure_fg, markers);
    // 8. 레이블링 된 마커 값에 1을 더하여 확실한 배경이 1이 되도록 함
    markers = markers + 1;
    // 9. 확실하지 않은 영역(unknown)은 0으로 설정
    markers.setTo(0, unknown == 255);

//    cv::imshow("markers", markers);

    // 10. 워터쉐드 알고리즘 적용
    cv::watershed(src, markers);

    // 11. 경계선 표시 (워터쉐드 결과가 -1인 픽셀은 경계)
    cv::Mat result = src.clone();
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            if (markers.at<int>(i, j) == -1)
            {
                result.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255); // 경계를 빨간색으로 표시
            }
        }
    }

    // 결과 이미지 출력
    cv::imshow("Watershed Result", result);
    cv::waitKey(0);
    return 0;
}
