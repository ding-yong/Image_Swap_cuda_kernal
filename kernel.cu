//kernel.cu
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include<math.h>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <ctime>
#include "opencv2/core/cuda.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;


//自定义内核函数
__global__ void swap_rb_kernel(const PtrStepSz<uchar3> src1, const PtrStepSz<uchar3> src2, PtrStep<uchar3> dst, int left_x, int right_x)
{
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	int i = threadIdx.y + blockIdx.y * blockDim.y;

	int src1pix = src1(i, j).x + src1(i, j).y + src1(i, j).z;
	int src2pix = src2(i, j).x + src2(i, j).y + src2(i, j).z;

	if (i >= 0 && j >= 0)
	{
		//if (j < left_x)
		if (src2pix == 0)
		{
			dst(i, j) = src1(i, j);
		}
		else if (src1pix == 0)//(j >= right_x)
		{
			dst(i, j) = src2(i, j);
		}
		else
		{
			//float d = (j - left_x) / (float)X;
			float srcLen = abs(j - left_x);
			float warpLen = abs(j - right_x);
			float d = srcLen / (srcLen + warpLen);

			dst(i, j).x = (uchar)(src1(i, j).x * (1 - d) + src2(i, j).x * d);
			dst(i, j).y = (uchar)(src1(i, j).y * (1 - d) + src2(i, j).y * d);
			dst(i, j).z = (uchar)(src1(i, j).z * (1 - d) + src2(i, j).z * d);
			//printf("HelloWorld! CPU %d %d %d\n", dst(i, j).x, dst(i, j).x, dst(i, j).x);


			//    dst(i, j) = (uchar)(src1(i, j) * (1 - d) + src2(i - Y, j - left)[k] * d);
		}
	}
}

void swap_rb_caller(const PtrStepSz<uchar3>& src, const PtrStepSz<uchar3>& src1, PtrStep<uchar3> dst, int left_x, int right_x, cudaStream_t stream)
{
	dim3 block(32, 8);
	dim3 grid((src.cols + block.x - 1) / block.x, (src.rows + block.y - 1) / block.y);

	swap_rb_kernel <<<grid, block, 0, stream>>>(src, src1, dst, left_x, right_x);
	if (stream == 0)
		cudaDeviceSynchronize();
}
