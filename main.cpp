// main.cpp
#include <iostream>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/flann/flann.hpp"
#include<opencv2/calib3d/calib3d.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/core/cuda.hpp"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/stitching.hpp>
#include<math.h>
#include<ctime>
using namespace cv::cuda;
using namespace std;
using namespace cv;

void swap_rb(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, int left_x, int right_x, Stream& stream = Stream::Null());
int main()
{
	Mat right = imread("D:\\image_data\\image1.jpg");
	Mat left = imread("D:\\image_data\\image2.jpg");
	namedWindow("src", WINDOW_NORMAL);
	namedWindow("gpu", WINDOW_NORMAL);
	int left_x = 313, right_x = 1079;

	imshow("src", right);
	clock_t start, finish;
	start = clock();
	GpuMat gpuRight, gpuLeft, output;
	gpuRight.upload(right);
	gpuLeft.upload(left);
	swap_rb(gpuRight, gpuLeft, output, left_x, right_x);
	Mat image;
	output.download(image);
	finish = clock();
	cout << "blend time:" << (finish - start) / CLOCKS_PER_SEC << "s" << endl;
	cout << "img.size:" << image.size() << endl;
	imwrite("D:\\image_data\\result.jpg", image);
	//imshow("gpu", image);
	//waitKey(0);
	return 0;

}
