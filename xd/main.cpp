#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


using namespace cv;


extern void FFTWrapper(const cv::Mat& in, cv::Mat& out);


void FSCPU(const cv::Mat& in, cv::Mat& out)
{
	const int p = 128;
	const int black = 0;
	const int white = 255;

	int width = in.cols;
	int heigth = in.rows;

	int* error = new int[width*heigth];



	for (int y = 0; y < heigth; y++)
	{
		for (int x = 0; x < width; x++)
		{
			error[y*width + x] = 0;
		}
	}



	for (int y = 0; y < in.rows; y++)
	{
		for (int x = 0; x < in.cols; x++)
		{
			int e = 0;
			if (in.data[y*in.cols + x] + error[y*width + x] < p)
			{
				out.data[y*out.cols + x] = black;
				e = in.data[y*in.cols + x] + error[y*width + x];
			}
			else
			{
				out.data[y*out.cols + x] = white;
				e = in.data[y*in.cols + x] + error[y*width + x] - 255;
			}

			if (x < width - 1)
			{
				error[y*width + x + 1] += e * 7 / 16;  //prawa

				if (y < heigth - 1)
				{
					error[(y + 1)*width + x + 1] += e * 1 / 16; //prawa dol
				}
			}

			if (y < heigth - 1)
			{
				error[(y + 1)*width + x] += e * 5 / 16;

				if (x > 0)
				{
					error[(y + 1)*width + x - 1] += e * 3 / 16;
				}
			}
		}
	}
	delete[]error;
}

int main(void)
{
	cv::Mat in;
	in = imread("C:\\Users\\Olga\\source\\repos\\xd\\xd\\vw.jpg", IMREAD_GRAYSCALE);

	cv::Mat out(in.rows, in.cols, CV_8UC1);

	FFTWrapper(in, out);


	//	FSCPU(in, out);



	namedWindow("Input", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Input", in);                   // Show our image inside it.

	cv::imwrite("spectrum_magnitude.jpg", out);

	namedWindow("Output", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Output", out);                   // Show our image inside it.

	waitKey(0);

	return 0;
}