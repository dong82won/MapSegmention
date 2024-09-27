#include "MSED.h"

MSED::MSED(void)
{
	//idxAll = 1;
}

MSED::~MSED(void)
{
}

// cv::Mat MSED::getThinningImg()
// {
// 	return thinningImg;
// }

double MSED::distancePoint2Point(const Point &p1, const Point &p2)
{
	return sqrt(powf(p1.x - p2.x, 2) + powf(p1.y - p2.y, 2));
}

uchar MSED::calculatePixelAverage(uchar *data, int coor_x, int coor_y, int winSize)
{

	int windowSum = 0;
	int windowPixels = 0;

	// 윈도우의 좌표 범위 계산
	int minX = max(0, coor_x - winSize / 2);
	int minY = max(0, coor_y - winSize / 2);
	int maxX = min(cols_1, coor_x + winSize / 2);
	int maxY = min(rows_1, coor_y + winSize / 2);

	// 윈도우 내의 픽셀 값 더하기
	for (int i = minY; i <= maxY; ++i)
	{
		for (int j = minX; j <= maxX; ++j)
		{
			int piexls = (int)data[i * cols + j];
			if (piexls != 0)
			{
				windowSum += piexls;
				windowPixels++;
			}
		}
	}

	// 평균값 계산
	uchar average = round(static_cast<double>(windowSum) / (double)windowPixels);
	// printf("x: %3d, y:%3d = %3d => avgpixel %3d\n", coor_x, coor_y, data[coor_y * cols + coor_x], average);
	return average;
}

// 인접한 포인터를 합치는 함수
std::vector<Point> MSED::mergeAdjacentPoints(const std::vector<Point> &points, double threshold)
{
	std::vector<Point> mergedPoints;
	std::vector<bool> visited(points.size(), false); // 각 포인터의 방문 여부를 추적

	for (int i = 0; i < points.size(); ++i)
	{
		if (!visited[i])
		{
			Point mergedPoint = points[i]; // 현재 포인터를 병합 대상으로 초기화
			int count = 1;				   // 현재 포인터를 포함한 개수

			for (int j = i + 1; j < points.size(); ++j)
			{
				if (!visited[j])
				{
					double dist = distancePoint2Point(points[i], points[j]);
					if (dist <= threshold)
					{
						// 인접한 포인터를 찾으면 병합 대상에 더하고 방문 플래그를 설정
						mergedPoint.x += points[j].x;
						mergedPoint.y += points[j].y;
						visited[j] = true;
						count++;
					}
				}
			}

			// 병합 대상의 포인터를 평균값으로 업데이트
			mergedPoint.x = (round)(mergedPoint.x / count);
			mergedPoint.y = (round)(mergedPoint.y / count);
			// 병합된 포인터를 결과 리스트에 추가
			mergedPoints.push_back(mergedPoint);
		}
	}

	return mergedPoints;
}

void MSED::MSEdge(uchar *map, int map_rows, int map_cols, std::vector<std::vector<cv::Point>> &edges)
{
	rows = map_rows;
	cols = map_cols;

	rows_1 = rows - 1;
	cols_1 = cols - 1;

	// edge map thinning
	cv::Mat imgMask(rows, cols, CV_8UC1, cv::Scalar(0));
	uchar *temp_map = imgMask.data;

	int imgSize = rows * cols;
	
	for (int i = 0; i < imgSize; ++i)
	{
		if (map[i] > 128)
		{
			*temp_map = 255;
		}
		temp_map++;
	}

	thinningGuoHall(imgMask);

	edgeTracking(imgMask, edges);
	edgeConnection(edges);
}

bool MSED::next(cv::Point &pt, uchar **ptrM, int &dir)
{
	if (pt.x < 1 || pt.x >= cols_1 || pt.y < 1 || pt.y >= rows_1)
	{
		return false;
	}

	for (int i = 0; i < idxSearch[dir].size(); ++i)
	{
		int dirIdx = idxSearch[dir][i];

		if (*(*ptrM + offsetTotal[dirIdx]))
		{
			for (int j = 0; j < idxSearch[dir].size(); ++j)
			{
				int dirIdxTemp = idxSearch[dir][j];

				// printf( " -------> %3d \n",*(*ptrM + offsetTotal[dirIdxTemp]));

				if (*(*ptrM + offsetTotal[dirIdxTemp]) == 1) // find a connect pixel
				{
					printf(" -------> find a connect pixel \n");
					return false;
				}
			}

			pt.x += offsetX[dirIdx];
			pt.y += offsetY[dirIdx];
			*ptrM += offsetTotal[dirIdx];

			dir = (dirIdx + 4) % 8;
			return true;
		}
	}
	return false;
}

void MSED::thinningGuoHall(cv::Mat &img)
{
	makeUpImage(img);

	img /= 255;
	cv::Mat prev(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));
	cv::Mat diff(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));

	int iteration = 0;

	do
	{
		imgThinning(img, 0);
		imgThinning(img, 1);

		cv::absdiff(img, prev, diff);
		img.copyTo(prev);
		iteration++;

	} while (cv::countNonZero(diff) > 0);

	cout << "iteration: " << iteration << endl;
	img *= 255;
	//thinningImg = img.clone();
}

void MSED::imgThinning(cv::Mat &img, int iter)
{
	// Code for thinning a binary image using Guo-Hall algorithm.
	/**
	 * Perform one thinning iteration.
	 * Normally you wouldn't call this function directly from your code.
	 *
	 * @param  im    Binary image with range = 0-1
	 * @param  iter  0=even, 1=odd
	 */

	int colsCur = img.cols;
	int rowsCur = img.rows;
	int colsCur_1 = colsCur - 1;
	int rowsCur_1 = rowsCur - 1;

	cv::Mat marker(rowsCur, colsCur, CV_8UC1, cv::Scalar::all(0));

	for (int i = 1; i < rowsCur_1; i++)
	{
		for (int j = 1; j < colsCur_1; j++)
		{
			// int marker = 0;
			uchar *ptr = img.data + i * img.cols + j;
			if (!*ptr)
			{
				continue;
			}

			uchar p2 = ptr[-colsCur];
			uchar p3 = ptr[-colsCur + 1];
			uchar p4 = ptr[1];
			uchar p5 = ptr[colsCur + 1];
			uchar p6 = ptr[colsCur];
			uchar p7 = ptr[colsCur - 1];
			uchar p8 = ptr[-1];
			uchar p9 = ptr[-colsCur - 1];

			int C = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) + (!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
			int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
			int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);

			int N = N1 < N2 ? N1 : N2;
			int m = iter == 0 ? ((p6 | p7 | !p9) & p8) : ((p2 | p3 | !p5) & p4);

			if (C == 1 && (N >= 2 && N <= 3) & m == 0)
				marker.at<uchar>(i, j) = 1;
		}
	}

	img &= ~marker;
}

void MSED::makeUpImage(cv::Mat &img)
{
	int xTemp[8] = {-1, 0, 1, 1, 1, 0, -1, -1};
	offsetX = std::vector<int>(xTemp, xTemp + 8);

	int yTemp[8] = {-1, -1, -1, 0, 1, 1, 1, 0};
	offsetY = std::vector<int>(yTemp, yTemp + 8);

	offsetTotal.resize(8);
	for (int i = 0; i < 8; ++i)
	{
		offsetTotal[i] = offsetY[i] * cols + offsetX[i];
	}

	for (int y = 1; y < rows_1; ++y)
	{
		for (int x = 1; x < cols_1; ++x)
		{
			uchar *ptr = img.data + y * img.cols + x;

			if (!*ptr)
			{
				int count = 0;
				for (int m = 0; m < 8; ++m)
				{
					if (*(ptr + offsetTotal[m]))
					{
						count++;
					}
				}

				if (count >= 8)
				{
					*ptr = 255;
				}
			}
		}
	}
}

void MSED::edgeTracking(cv::Mat &mask, std::vector<std::vector<cv::Point>> &edgeChains)
{

	int xTemp[8] = {-1, 0, 1, 1, 1, 0, -1, -1};
	offsetX = std::vector<int>(xTemp, xTemp + 8);

	int yTemp[8] = {-1, -1, -1, 0, 1, 1, 1, 0};
	offsetY = std::vector<int>(yTemp, yTemp + 8);

	offsetTotal.resize(8);
	for (int i = 0; i < 8; ++i)
	{
		offsetTotal[i] = offsetY[i] * cols + offsetX[i];
	}

	idxSearch = std::vector<std::vector<int>>(8, std::vector<int>(5, 0));
	for (int i = 0; i < 8; ++i)
	{
		idxSearch[i][0] = (i + 4) % 8;
		idxSearch[i][1] = (i + 3) % 8;
		idxSearch[i][2] = (i + 5) % 8;
		idxSearch[i][3] = (i + 2) % 8;
		idxSearch[i][4] = (i + 6) % 8;
	}

	cout << "rows: " << rows << " ,cols: " << cols << endl;

	for (int y = 0; y < rows; ++y)
	{
		for (int x = 0; x < cols; ++x)
		{
			int loc = y * cols + x;
			if (!mask.data[loc])
			{
				continue;
			}

			int dir1 = 0, dir2 = 0;

			// if ( OmapAll.data[loc] == 0 )
			if (mask.data[loc] == 0) // vertical edge
			{
				dir1 = 1;
				dir2 = 5;
			}
			else
			{
				dir1 = 3;
				dir2 = 7;
			}

			std::vector<cv::Point> chain;
			cv::Point pt(x, y);
			uchar *ptrM = mask.data + loc;
			do
			{
				chain.push_back(pt);
				*ptrM = 0;
			} while (next(pt, &ptrM, dir1));

			cv::Point temp;
			for (int m = 0, n = chain.size() - 1; m < n; ++m, --n)
			{
				temp = chain[m];
				chain[m] = chain[n];
				chain[n] = temp;
			}

			// Find and add feature pixels to the begin of the string.
			pt.x = x;
			pt.y = y;
			ptrM = mask.data + loc;
			if (next(pt, &ptrM, dir2))
			{
				do
				{
					chain.push_back(pt);
					*ptrM = 0;
				} while (next(pt, &ptrM, dir2));
			}

			if (chain.size() >= 10)
			{
				edgeChains.push_back(chain);
			}
		}
	}
}

void MSED::edgeConnection(std::vector<std::vector<cv::Point>> &edgeChains)
{
	cv::Mat maskTemp(rows, cols, CV_64FC1, cv::Scalar(-1));
	double *ptrM = (double *)maskTemp.data;

	for (int i = 0; i < edgeChains.size(); ++i)
	{
		for (int j = 0; j < edgeChains[i].size(); ++j)
		{
			int loc = edgeChains[i][j].y * cols + edgeChains[i][j].x;
			ptrM[loc] = i;
		}
	}

	int step = 8;
	std::vector<int> mergedIdx(edgeChains.size(), 0);
	for (int i = 0; i < edgeChains.size(); ++i)
	{
		if (mergedIdx[i])
		{
			continue;
		}

		if (i == 8)
		{
			int aa = 0;
		}

		bool merged = false;
		int idxChain = 0, idxPixelStart = 0, idxPixelEnd = 0;

		if (i == 8)
		{
			int aa = 0;
		}
		// from the begin
		int t1 = min(step, (int)edgeChains[i].size());
		int j = 0;
		for (j = 0; j < t1; ++j)
		{
			int x = edgeChains[i][j].x;
			int y = edgeChains[i][j].y;
			if (x < 1 || x >= cols_1 || y < 1 || y >= rows_1)
			{
				continue;
			}

			double *ptrTemp = (double *)maskTemp.data + y * cols + x;

			for (int m = 0; m < 8; ++m)
			{
				cout << "offsetX[m]: " << offsetX[m] << endl;
			}

			for (int m = 0; m < 8; ++m)
			{
				int idxSearched = *(ptrTemp + offsetTotal[m]);

				if (idxSearched >= 0 && idxSearched != i)
				{
					if (mergedIdx[idxSearched])
					{
						continue;
					}

					int n = 0;
					int xSearched = x + offsetX[m];
					int ySearched = y + offsetY[m];
					for (n = 0; n < edgeChains[idxSearched].size(); ++n)
					{
						if (edgeChains[idxSearched][n].x == xSearched && edgeChains[idxSearched][n].y == ySearched)
						{
							break;
						}
					}

					if (n < step || n > edgeChains[idxSearched].size() - step) // merge these two edge chain
					{
						merged = true;
						idxChain = idxSearched;
						if (n < step)
						{
							idxPixelStart = n;
							idxPixelEnd = edgeChains[idxSearched].size();
						}
						else
						{
							idxPixelStart = n;
							idxPixelEnd = 0;
						}
						break;
					}
				}
			}

			if (merged)
			{
				break;
			}
		}

		if (merged)
		{
			std::vector<cv::Point> mergedChain;
			for (int m = edgeChains[i].size() - 1; m >= j; --m)
			{
				mergedChain.push_back(edgeChains[i][m]);
			}

			int order = 1;
			if (idxPixelEnd < idxPixelStart)
			{
				order = -1;
			}

			for (int m = idxPixelStart; m != idxPixelEnd; m += order)
			{
				mergedChain.push_back(edgeChains[idxChain][m]);
			}
			edgeChains.push_back(mergedChain);

			for (int m = 0; m < j; ++m)
			{
				int loc = edgeChains[i][m].y * cols + edgeChains[i][m].x;
				ptrM[loc] = -1;
			}

			for (int m = idxPixelStart; m != idxPixelEnd; m += order)
			{
				int loc = edgeChains[idxChain][m].y * cols + edgeChains[idxChain][m].x;
				ptrM[loc] = -1;
			}

			int idxTemp = edgeChains.size() - 1;
			for (int m = 0; m < mergedChain.size(); ++m)
			{
				int loc = mergedChain[m].y * cols + mergedChain[m].x;
				ptrM[loc] = idxTemp;
			}

			mergedIdx[i] = 1;
			mergedIdx[idxChain] = 1;
			mergedIdx.push_back(0);
			continue;
		}

		// from the end
		int t2 = max(0, (int)edgeChains[i].size() - 1 - step);
		for (j = edgeChains[i].size() - 1; j > t2; --j)
		{
			int x = edgeChains[i][j].x;
			int y = edgeChains[i][j].y;
			if (x < 1 || x >= cols_1 || y < 1 || y >= rows_1)
			{
				continue;
			}

			double *ptrTemp = (double *)maskTemp.data + y * cols + x;
			for (int m = 0; m < 8; ++m)
			{
				int idxSearched = *(ptrTemp + offsetTotal[m]);

				if (idxSearched >= 0 && idxSearched != i)
				{
					if (mergedIdx[idxSearched])
					{
						continue;
					}

					int n = 0;
					int xSearched = x + offsetX[m];
					int ySearched = y + offsetY[m];
					for (n = 0; n < edgeChains[idxSearched].size(); ++n)
					{
						if (edgeChains[idxSearched][n].x == xSearched && edgeChains[idxSearched][n].y == ySearched)
						{
							break;
						}
					}

					if (n < step || n > edgeChains[idxSearched].size() - step) // merge these two edge chain
					{
						merged = true;
						idxChain = idxSearched;
						if (n < step)
						{
							idxPixelStart = n;
							idxPixelEnd = edgeChains[idxSearched].size();
						}
						else
						{
							idxPixelStart = n;
							idxPixelEnd = 0;
						}
						break;
					}
				}
			}

			if (merged)
			{
				break;
			}
		}

		if (merged)
		{
			std::vector<cv::Point> mergedChain(edgeChains[i].begin(), edgeChains[i].begin() + j);

			int order = 1;
			if (idxPixelEnd < idxPixelStart)
			{
				order = -1;
			}

			for (int m = idxPixelStart; m != idxPixelEnd; m += order)
			{
				mergedChain.push_back(edgeChains[idxChain][m]);
			}
			edgeChains.push_back(mergedChain);

			for (int m = j; m < edgeChains[i].size(); ++m)
			{
				int loc = edgeChains[i][m].y * cols + edgeChains[i][m].x;
				ptrM[loc] = -1;
			}

			for (int m = idxPixelStart; m != idxPixelEnd; m += order)
			{
				int loc = edgeChains[idxChain][m].y * cols + edgeChains[idxChain][m].x;
				ptrM[loc] = -1;
			}

			int idxTemp = edgeChains.size() - 1;
			for (int m = 0; m < mergedChain.size(); ++m)
			{
				int loc = mergedChain[m].y * cols + mergedChain[m].x;
				ptrM[loc] = idxTemp;
			}

			mergedIdx[i] = 1;
			mergedIdx[idxChain] = 1;
			mergedIdx.push_back(0);
			continue;
		}
	}

	std::vector<std::vector<cv::Point>> edgeChainsNew;
	for (int i = 0; i < mergedIdx.size(); ++i)
	{
		if (!mergedIdx[i])
		{
			edgeChainsNew.push_back(edgeChains[i]);
		}
	}
	edgeChains = edgeChainsNew;
}
