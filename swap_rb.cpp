// swap_rb.cpp
#include <iostream>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/flann/flann.hpp"
#include<opencv2/calib3d/calib3d.hpp>
#include<opencv2/opencv.hpp>
#include "opencv2/core/cuda.hpp"
#include <opencv2/core/cuda_stream_accessor.hpp>

using namespace cv::cuda;
using namespace std;
using namespace cv;

#include <iostream>
#include <opencv2/opencv.hpp>
void swap_rb_caller(const PtrStepSz<uchar3>& src, const PtrStepSz<uchar3>& src1, PtrStep<uchar3> dst, int left_x, int right_x, cudaStream_t stream);
void swap_rb(const GpuMat& src, const GpuMat& src1, GpuMat& dst, int left_x, int right_x, Stream& stream = Stream::Null())
{
	CV_Assert(src.type() == CV_8UC3);
	dst.create(src.size(), src.type());
	cudaStream_t s = StreamAccessor::getStream(stream);
	swap_rb_caller(src, src1, dst, left_x, right_x, s);
}
