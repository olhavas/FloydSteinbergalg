
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/mat.hpp>
#include <iostream>
#include <vector>

constexpr auto SIZE = 16;
#define POSITION (((blockIdx.x*SIZE+y)*blockDim.x)*SIZE) + (threadIdx.x*SIZE) + x


__global__ void multiplication(float *A, float* B, float *C, int N) {
	int ROW = blockIdx.y*blockDim.y + threadIdx.y;
	int COL = blockIdx.x*blockDim.x + threadIdx.x;

	if (ROW < N && COL < N) {

		float tmpSum = 0.0;
		for (int i = 0; i < N; i++) {
			tmpSum += A[ROW * N + i] * B[i * N + COL];
		}
		C[ROW * N + COL] = tmpSum;
	}



}
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

}

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