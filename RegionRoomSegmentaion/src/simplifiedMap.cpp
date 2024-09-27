#include "simplifiedMap.h"
 

SimplifyMap::SimplifyMap() 
{ 
}

SimplifyMap::~SimplifyMap()
{
}

// cv::Mat SimplifyMap::getImgColor()
// {
// 	return m_img_color;
// }

// cv::Mat SimplifyMap::getSimplifiedMap()
// {
// 	return m_img_gray;
// }


// void SimplifyMap::initializationSimpliedMap(cv::Mat img_gray)
// { 
// 	m_img_gray = img_gray; 
// 	cv::Mat binary;
// 	cv::threshold(img_gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
// 	cv::Mat gradient;
// 	cv::morphologyEx(binary, gradient, cv::MORPH_GRADIENT, cv::Mat());

// 	cv::Mat inv_gradient;
// 	bitwise_not(gradient, inv_gradient);
// 	m_img_gradient = inv_gradient.clone(); 

// 	//cvtColor(img_gray, m_img_color, cv::COLOR_GRAY2BGR);  
// }


cv::Mat SimplifyMap::makeImgGradient(cv::Mat img_gray)
{

	cv::Mat binary;
	cv::threshold(img_gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	cv::Mat gradient;
	cv::morphologyEx(binary, gradient, cv::MORPH_GRADIENT, cv::Mat()); 
	//cv::morphologyEx(gradient, gradient, cv::MORPH_OPEN, cv::Mat());


	cv::Mat inv_gradient;
	bitwise_not(gradient, inv_gradient); 

	return inv_gradient;
}


double SimplifyMap::converterMean(cv::Mat img_box)
{
	double sum = 0.0;
	int num = 0;
	for (int i = 0; i < img_box.rows; i++)
	{
		uchar *prt = img_box.ptr<uchar>(i);
		for (int j = 0; j < img_box.cols; j++)
		{
			if (prt[j] != 0)
			{
				sum += (double)prt[j];
				num++;
			}
		}
	}

	double mean;
	if (num != 0)
	{
		mean = sum / (double)num;
	}
	else
	{
		mean = 0.0;
	}
	return mean;
}

int SimplifyMap::normalizeMat2Percent(cv::Mat m_arry)
{ 	
	 
	//printf("m_arry: rows %d, cols: %3d\n", m_arry.rows, m_arry.cols);
	cv::Mat m_norm = cv::Mat::zeros(m_arry.rows, m_arry.cols, CV_8UC1); 
	for (int i = 0; i < m_arry.rows; ++i)
	{
		uchar *prt = m_arry.ptr<uchar>(i);
		uchar *prt_norm = m_norm.ptr<uchar>(i);  

		for (int j = 0; j < m_arry.cols; ++j)
		{ 
			prt_norm[j] = (uchar)(((double)prt[j] / 255.0) * 100.0);
			//printf("%3d\t", prt_norm[j]);
		}
		//printf("\n");
	}

	// printf("\n\n");
	// for (int i = 0; i < m_arry.rows; ++i)
	// {
	// 	uchar *prt = m_arry.ptr<uchar>(i);  
	// 	for (int j = 0; j < m_arry.cols; ++j)
	// 	{ 			 
	// 		printf("%3d\t", prt[j]);
	// 	}
	// 	printf("\n");
	// } 


	// printf("\n");
	// cv::imshow("test", m_norm);
	// cv::waitKey(0);

	//int mean = (int)converterMean(m_norm);
	cv::Scalar meanVal = mean(m_norm); 
	return (int)meanVal[0]; //mean;
}


// double SimplifyMap::normalizeMat2Percent(cv::Mat m_arry)
// {
// 	// // Find the minimum and maximum values
// 	// double minVal, maxVal;
// 	// cv::Point minLoc, maxLoc;
// 	// cv::minMaxLoc(m_arry, &minVal, &maxVal, &minLoc, &maxLoc);

// 	// printf("size: %3d x %3d\n", m_arry.rows, m_arry.cols);
// 	// printf("%3.3f, %3.3f\n", minVal, maxVal);


// 	cv::Scalar meanVal = mean(m_arry);
// 	std::cout << meanVal[0] << std::endl;

// 	cv::Mat m_norm = cv::Mat::zeros(m_arry.rows, m_arry.cols, CV_8UC1);
// 	for (int i = 0; i < m_arry.rows; i++)
// 	{
// 		uchar *prt_norm = m_norm.ptr<uchar>(i);
// 		//uchar *prt = m_arry.ptr<uchar>(i);
// 		for (int j = 0; j < m_arry.cols; j++)
// 		{
// 			//double norm = (((double)prt[j] - minVal) / (maxVal - minVal)) * 100;
// 			//printf("norm: %3.3f\t", norm);
// 			prt_norm[j] = (uchar)round(meanVal[0]);
// 		}
// 		//printf("\n");
// 	}
// 	double mean = converterMean(m_norm);
// 	return meanVal[0];//mean;
// }

std::vector<std::vector<BOX>> SimplifyMap::blockification(cv::Mat img_gray)
{
	int box_size = 3;

	int rows = img_gray.rows;
	int cols = img_gray.cols;

	int rows_pad = rows % box_size;
	int cols_pad = cols % box_size;

	int step_rows = rows - rows_pad;
	int step_cols = cols - cols_pad;

	int box_rows = (step_rows / box_size) - 1;
	int box_cols = (step_cols / box_size) - 1;

	std::vector<cv::Mat> img_box;
	std::vector<cv::Rect> rect_box;

	for (int i = box_size; i < step_rows; i += box_size)
	{
		for (int j = box_size; j < step_cols; j += box_size)
		{
			cv::Rect box_roi(j, i, box_size, box_size);
			cv::Mat img_ = img_gray(box_roi).clone();

			rect_box.push_back(box_roi);
			img_box.push_back(img_);
		}
	}

	std::vector<std::vector<BOX>> info_box;
	// cv::imshow("color_img_gray", color_img_gray);
	for (int i = 0; i < box_rows; i++)
	{
		std::vector<BOX> boxs;
		BOX tmp;
		int base = 50;
		for (int j = 0; j < box_cols; j++)
		{
			int idx = (i * box_cols) + j;
			int box_norm = normalizeMat2Percent(img_box[idx]);

			//printf("box_norm : %3d\t", box_norm);

			if ( (( base + 3) >= box_norm) && ((base - 3) <= box_norm ))
			//if ((box_norm <= 45.0 + 3.0) && (box_norm >= 45.0 - 3.0))
			{ 
				tmp.state = BACKGROUND;
				tmp.roi = rect_box[idx];
				boxs.push_back(tmp);
			}
			else
			{
				if (box_norm >= base)
				{ 
					tmp.state = DRIVING;
					tmp.roi = rect_box[idx];
					boxs.push_back(tmp);
				}
				else
				{ 
					tmp.state = WALL;
					tmp.roi = rect_box[idx];
					boxs.push_back(tmp);
				}
			} 
		} 
		info_box.push_back(boxs);
	} 
	return info_box;
}

bool SimplifyMap::wallLineMap(cv::Mat img_box, int type, int threshold_pixel)
{
	int piexl_num = 0;
	for (int i = 0; i < img_box.rows; i++)
	{
		uchar *prt = img_box.ptr<uchar>(i);
		for (int j = 0; j < img_box.cols; j++)
		{
			uchar data = prt[j]; 
			if (type == 0)
			{
				if (data == 0)
				{
					piexl_num++;
				}
			}
			else if (type == 1)
			{
				if (data < 64) // 255 에서 64...(이미지 그레이 기준)
				{
					piexl_num++;
				}
			}
		}
	}
	if (piexl_num >= threshold_pixel)
		return true;
	else
		return false;
}

void SimplifyMap::checkBlockWall(std::vector<std::vector<BOX>> &info_box, cv::Mat img_gray, cv::Mat img_gradient)
{

	cv::imshow("check_block_img_gray", img_gray);
	cv::imshow("check_block_img_gradient", img_gradient);

	for (int i = 0; i < info_box.size(); i++)
	{
		for (int j = 0; j < info_box[i].size(); j++)
		{
			cv::Rect roi = info_box[i][j].roi;
			int center_sate = info_box[i][j].state;

			cv::Mat box_gradient = img_gradient(roi).clone();
			if (center_sate == BACKGROUND)
			{
				if (wallLineMap(box_gradient, 0, 3))
					info_box[i][j].state = WALL;
			}

			cv::Mat box_gray = img_gray(roi).clone();
			if (center_sate == DRIVING)
			{
				if (wallLineMap(box_gray, 1, 3))
					info_box[i][j].state = WALL;
			}
		}
	}
}

void SimplifyMap::checkUndriving(std::vector<std::vector<BOXINFO>> &pixelBox)
{
	for (int i = 0; i < pixelBox.size(); i++)
	{
		for (int j = 0; j < pixelBox[i].size(); j++)
		{
			cv::Rect roi = pixelBox[i][j].info.roi;
			int center_sate = pixelBox[i][j].info.state;

			if (center_sate == DRIVING)
			{
				int backgroad = 0;
				for (int d = 0; d < pixelBox[i][j].dir.size(); d++)
				{
					int dir_state = pixelBox[i][j].dir[d].second;
					if (dir_state == BACKGROUND)
					{
						backgroad++;
					}
				}

				if (backgroad > 1)
					pixelBox[i][j].info.state = UNDRIVING;
			}
		}
	}
}

std::vector<std::vector<BOXINFO>> SimplifyMap::get8Neighbors(std::vector<std::vector<BOX>> info_box)
{

	// 8방향의 상대적인 좌표
	std::vector<std::pair<int, int>> directions = {
		{-1, -1}, {-1, 0}, {-1, 1}, 
		{0,  -1}, { 0, 0}, { 0, 1}, 
		{1,  -1}, { 1, 0}, { 1, 1}};

	//------------------------------------------------------------------------------
	std::vector<std::vector<BOXINFO>> pixelBox;
	for (int i = 0; i < info_box.size(); i++)
	{
		std::vector<BOXINFO> mv_dir;
		for (int j = 0; j < info_box[i].size(); j++)
		{
			BOXINFO tmp_dir;
			int center_sate = info_box[i][j].state;
			// printf("BOX ------- [%3d][%3d], center_sate = %d \n", i, j, center_sate);
			tmp_dir.info.roi = info_box[i][j].roi;
			tmp_dir.info.state = info_box[i][j].state;

			// std::vector<std::pair<int, int>> dir_info;
			for (int d = 0; d < directions.size(); d++)
			{
				int dy = directions[d].first;
				int dx = directions[d].second;
				int ny = i + dy;
				int nx = j + dx;

				std::pair<int, int> ds;
				if (nx >= 0 && nx < info_box[i].size() && ny >= 0 && ny < info_box.size())
				{
					// rectangle(color_img_gray, Rect(info_box[i][j].roi), CV_RGB(255, 255, 255));
					int state = info_box[ny][nx].state;
					// printf("state[%d] %d\n", d, state);
					ds.first = d;
					ds.second = state;
					// dir_info.push_back(ds);
					tmp_dir.dir.push_back(ds);
				}
			}
			mv_dir.push_back(tmp_dir);
		}
		pixelBox.push_back(mv_dir);
	}
 
	checkUndriving(pixelBox); 

	return pixelBox;
}

cv::Mat SimplifyMap::runSimplify(cv::Mat img_gray)
{ 	
	imshow("img_gray", img_gray);
	cv::Mat img_gradient = makeImgGradient(img_gray); 
	std::vector<std::vector<BOX>> info_box = blockification(img_gray); 
	
    checkBlockWall(info_box, img_gray, img_gradient);     
    std::vector<std::vector<BOXINFO>> simplify_info = get8Neighbors(info_box); 

	cv::Mat simpliedmap = makeSimplifiedImage(simplify_info, img_gray); 
	//cv::Mat simpliedmap = makeSimplifiedImageColor(simplify_info, img_gray); 

	return simpliedmap; 

}


cv::Mat SimplifyMap::makeSimplifiedImageColor(std::vector<std::vector<BOXINFO>> info, cv::Mat img_gray)
{

	cv::Mat img_base(img_gray.rows, img_gray.cols, CV_8UC3, CV_RGB(128, 128, 128));  

	//cv::Mat img_base = cv::Mat::zeros(img_gray.rows, img_gray.cols, CV_8UC1, CV_RGB(128, 128, 128));  

	for (int i = 0; i < info.size(); i++)
	{
		for (int j = 0; j < info[i].size(); j++)
		{
			cv::Rect roi = info[i][j].info.roi;
			int state = info[i][j].info.state;

			switch (state)
			{
			case BACKGROUND:
				rectangle(img_base, roi, CV_RGB(128, 128, 128), -1);
				// rectangle(binary_color, roi, CV_RGB(255, 0, 0));
				//  rectangle(inv_gradient_color, roi, CV_RGB(255, 0, 0));
				break;

			case DRIVING:
				rectangle(img_base, roi, CV_RGB(255, 255, 255), -1);
				// rectangle(binary_color, roi, CV_RGB(0, 255, 0));
				//  rectangle(inv_gradient_color, roi, CV_RGB(0, 255, 0));
				break;

			case WALL:
				rectangle(img_base, roi, CV_RGB(0, 0, 0), -1);
				// rectangle(binary_color, roi, CV_RGB(0, 0, 255));
				//  rectangle(inv_gradient_color, roi, CV_RGB(0, 0, 255));
				break;

			case UNDRIVING:
				rectangle(img_base, roi, CV_RGB(0, 255, 255), -1);
				// rectangle(binary_color, roi, CV_RGB(255, 255, 255));
				//  rectangle(inv_gradient_color, roi, CV_RGB(255, 255, 255));
				break;

			default:
				break;
			}
		}
	}
	return img_base; 
}


cv::Mat SimplifyMap::makeSimplifiedImage(std::vector<std::vector<BOXINFO>> info, cv::Mat img_gray)
{
	//cv::Mat img_base(img_gray.rows, img_gray.cols, CV_8UC1, cv::Scalar(128));  
	cv::Mat img_base(img_gray.rows, img_gray.cols, CV_8UC1, cv::Scalar(0));  


	for (int i = 0; i < info.size(); i++)
	{ 
		for (int j = 0; j < info[i].size(); j++)
		{
			//printf("info.size(%3d): %3d\n", info.size(), info[i].size());
			
			cv::Rect roi = info[i][j].info.roi;
			int state = info[i][j].info.state;

			switch (state)
			{
			// case BACKGROUND:
			// 	rectangle(img_base, roi, cv::Scalar(128), -1);
			// 	// rectangle(binary_color, roi, CV_RGB(255, 0, 0));
			// 	//  rectangle(inv_gradient_color, roi, CV_RGB(255, 0, 0));
			// 	break;

			case DRIVING:
				rectangle(img_base, roi, cv::Scalar(255), -1);
				// rectangle(binary_color, roi, CV_RGB(0, 255, 0));
				//  rectangle(inv_gradient_color, roi, CV_RGB(0, 255, 0));
				break;

			case WALL:
				rectangle(img_base, roi, cv::Scalar(0), -1);
				// rectangle(binary_color, roi, CV_RGB(0, 0, 255));
				//  rectangle(inv_gradient_color, roi, CV_RGB(0, 0, 255));
				break;

			case UNDRIVING:
				//rectangle(img_base, roi, cv::Scalar(64), -1);
				rectangle(img_base, roi, cv::Scalar(255), -1);
				// rectangle(binary_color, roi, CV_RGB(255, 255, 255));
				//  rectangle(inv_gradient_color, roi, CV_RGB(255, 255, 255));
				break;

			// default:
			// 	rectangle(img_base, roi, cv::Scalar(128), -1);
			// 	break;
			}
		}
	} 

	return img_base;
}