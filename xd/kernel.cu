#include <stdio.h>    
#include <cuda_runtime.h> 
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cufft.h>


//constexpr auto SIZE = 16;
#define M_PI 3.14159265358979323846 
//#define POSITION (((blockIdx.x*SIZE+y)*blockDim.x)*SIZE) + (threadIdx.x*SIZE) + x








void FFTWrapper(const cv::Mat& in, cv::Mat& out)
{	
	cv::Mat_<float> img(in);
	
	float *h_DataReal = img.ptr<float>(0);
	cufftComplex *h_DataComplex;

	// Image dimensions
	const int dataH = img.rows;
	const int dataW = img.cols;

	// Convert real input to complex
	h_DataComplex = (cufftComplex *)malloc(dataH * dataW * sizeof(cufftComplex));
	for (int i = 0; i < dataH*dataW; i++) {
		h_DataComplex[i].x = h_DataReal[i];
		h_DataComplex[i].y = 0.0f;
	}

	// Complex device pointers
	cufftComplex
		*d_Data,
		*d_DataSpectrum,
		*d_Result,
		*h_Result;

	// Plans for cuFFT execution
	cufftHandle
		fftPlanFwd;
		//fftPlanInv;

	// Allocate memory
	h_Result = (cufftComplex *)malloc(dataH    * dataW * sizeof(cufftComplex));
	cudaMalloc((void **)&d_DataSpectrum, dataH * dataW * sizeof(cufftComplex));
	cudaMalloc((void **)&d_Data, dataH   * dataW * sizeof(cufftComplex));
	cudaMalloc((void **)&d_Result, dataH * dataW * sizeof(cufftComplex));


	// Copy image to GPU
	cudaMemcpy(d_Data, h_DataComplex, dataH   * dataW * sizeof(cufftComplex), cudaMemcpyHostToDevice);


	// Forward FFT
	cufftPlan2d(&fftPlanFwd, dataH, dataW, CUFFT_C2C);
	cufftExecC2C(fftPlanFwd, (cufftComplex *)d_Data, (cufftComplex *)d_Result, CUFFT_FORWARD);

	// Inverse FFT
	//cufftPlan2d(&fftPlanInv, dataH, dataW, CUFFT_C2C);
	//cufftExecC2C(fftPlanInv, (cufftComplex *)d_DataSpectrum, (cufftComplex *)d_Result, CUFFT_INVERSE);

	// Copy result to host memory
	cudaMemcpy(h_Result, d_Result, dataH * dataW * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	
	

	// Convert cufftComplex to OpenCV real and imag Mat
	cv::Mat_<float> resultReal = cv::Mat_<float>(dataH, dataW);
	cv::Mat_<float> resultImag = cv::Mat_<float>(dataH, dataW);
	for (int i = 0; i < dataH; i++) {
		float* rowPtrReal = resultReal.ptr<float>(i);
		float* rowPtrImag = resultImag.ptr<float>(i);
		for (int j = 0; j < dataW; j++) {
			rowPtrReal[j] = h_Result[i*dataW + j].x / (dataH*dataW);
			rowPtrImag[j] = h_Result[i*dataW + j].y / (dataH*dataW);
		}
	}

	cv::Mat_<float> resultPhase;
	phase(resultReal, resultImag, resultPhase);


	cv::subtract(resultPhase, 2 * M_PI, resultPhase, (resultPhase > M_PI));
	resultPhase = ((resultPhase + M_PI) * 255) / (2 * M_PI);
	cv::Mat_<uchar> normalized = cv::Mat_<uchar>(dataH, dataW);
	resultPhase.convertTo(normalized, CV_8U);
	// Save phase image
	cv::imwrite("cuda_propagation_phase.png", resultPhase);

	
	cufftDestroy(fftPlanFwd);
	//cufftDestroy(fftPlanInv);
	cudaFree(d_DataSpectrum);
	cudaFree(d_Data);
	cudaFree(d_Result);
	
	
	// Save phase image


	// Compute amplitude and normalize to 8 bit
	cv::Mat_<float> resultAmplitude;
	magnitude(resultReal, resultImag, resultAmplitude);
	cv::Mat_<uchar> normalizedAmplitude = cv::Mat_<uchar>(dataH, dataW);
	resultAmplitude.convertTo(normalizedAmplitude, CV_8U);
	// Save phase image
	cv::imwrite("cuda_propagation_amplitude.png", resultAmplitude);

	cv::Mat magI(resultAmplitude);


	//magI += cv::Scalar::all(1);                    // switch to logarithmic scale
	cv::log(magI, magI);

	// crop the spectrum, if it has an odd number of rows or columns
	magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));

	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
	cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
	cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right
	cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	cv::normalize(magI, magI, 0, 1, cv::NormTypes::NORM_MINMAX); // Transform the matrix with float values into a
											// viewable image form (float between values 0 and 1).
	
	out = magI.clone();
}

/*
__global__
void FloydSteinberg(unsigned char* input, unsigned char* output, char* error, unsigned int rows, unsigned int cols)
{

	const int p = 128;
	const int black = 0;
	const int white = 255;

	int e = 0;

	for (int y = 0; y < SIZE; y++)
	{
		for (int x = 0; x < SIZE; x++)
		{
			if (input[POSITION] + error[POSITION] < p)
			{
				output[POSITION] = black;
				e = input[POSITION] + error[POSITION];
			}
			else
			{
				output[POSITION] = white;
				e = input[POSITION] + error[POSITION] - white;
			}




			if (x < SIZE - 1)
			{
				error[POSITION + 1] += e * 7 / 16;


				if (y < SIZE - 1)
				{
					error[(((blockIdx.x*SIZE + y + 1)*blockDim.x)*SIZE) + (threadIdx.x*SIZE) + x + 1] += e * 1 / 16;

				}
			}

			if (y < SIZE - 1)
			{
				error[(((blockIdx.x*SIZE + y + 1)*blockDim.x)*SIZE) + (threadIdx.x*SIZE) + x] += e * 5 / 16;

				{
					error[(((blockIdx.x*SIZE + y + 1)*blockDim.x)*SIZE) + (threadIdx.x*SIZE) + x - 1] += e * 3 / 16;

				}
			}


		}
	}






}



__global__
void FloydSteinbergST(unsigned char* input, unsigned char* output, char* error, unsigned int rows, unsigned int cols)
{
	const int p = 128;
	const int black = 0;
	const int white = 255;
	int width = cols;
	int heigth = rows;



	for (int y = 0; y < rows; y++)
	{
		for (int x = 0; x < cols; x++)
		{
			int e = 0;
			if (input[y*cols + x] + error[y*width + x] < p)
			{
				output[y*cols + x] = black;
				e = input[y*cols + x] + error[y*width + x];
			}
			else
			{
				output[y*cols + x] = white;
				e = input[y*cols + x] + error[y*width + x] - white;
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

}*/
/*
void FloydSteinbergWrapper(const cv::Mat& in, cv::Mat& out)
{

	unsigned char *input_prt, *output_ptr;
	char *error_ptr;

	cudaMalloc<unsigned char>(&input_prt, in.rows*in.cols);
	cudaMalloc<unsigned char>(&output_ptr, out.rows*out.cols);
	cudaMalloc<char>(&error_ptr, in.cols*in.rows);

	cudaMemcpy(input_prt, in.ptr(), in.rows*in.cols, cudaMemcpyHostToDevice);


	int blockSize = in.cols / SIZE;



	//FloydSteinberg << <in.rows, in.cols >> > (input_prt, output_ptr, error_ptr, in.rows, in.cols);
	FloydSteinberg << <blockSize, blockSize >> > (input_prt, output_ptr, error_ptr, in.rows, in.cols);

	cudaDeviceSynchronize();
	cudaMemcpy(out.ptr(), output_ptr, out.cols*out.rows, cudaMemcpyDeviceToHost);

	// Free memory
	cudaFree(input_prt);
	cudaFree(output_ptr);
	cudaFree(error_ptr);

}


void MatrixMulKernel(const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C) {
	int *inputA_prt, *inputB_prt, *output_ptr;



}
*/