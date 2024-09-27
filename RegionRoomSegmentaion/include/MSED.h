#ifndef _MSED_H_
#define _MSED_H_

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

#define MIN_FLOAT 0.000101
#define INF 1001001000111


class MSED
{
public:
	MSED(void);
	~MSED(void);

public:
	// my add fun()
	//cv::Mat getThinningImg();
	std::vector<cv::Point> mergeAdjacentPoints(const std::vector<cv::Point> &points, double threshold);

	uchar calculatePixelAverage(uchar *data, int coor_x, int coor_y, int winSize);

	void MSEdge(uchar *map, int map_rows, int map_cols, std::vector<std::vector<cv::Point>> &edges);
	void thinningGuoHall(cv::Mat &img);
	void edgeTracking(cv::Mat &edgeMap, std::vector<std::vector<cv::Point>> &edgeChains);
	void edgeConnection( std::vector<std::vector<cv::Point> > &edgeChains );

	uchar *m_map;

private:
	double distancePoint2Point(const Point &p1, const Point &p2);

	void makeUpImage(cv::Mat &img);
	void imgThinning(cv::Mat &img, int iter);
	bool next(cv::Point &pt, uchar **ptrM, int &dir);

	//int idxAll;
	int rows, cols, rows_1, cols_1;
	// cv::Mat EmapAll, OmapAll;
	//cv::Mat thinningImg;
	std::vector<int> offsetX, offsetY, offsetTotal;
	std::vector<std::vector<int>> idxSearch;
	cv::Mat maskMakeUp;
};

#endif //_MSED_H_
