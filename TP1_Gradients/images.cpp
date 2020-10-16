#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;

Mat float2byte(const Mat& If)
{
	double minVal, maxVal;
	minMaxLoc(If, &minVal, &maxVal);
	Mat Ib;
	If.convertTo(Ib, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
	return Ib;
}

// Do not try to understand if do not know pointers. Will be seen during lesson #2
void onTrackbar(int sigma, void* p)
{
	Mat A = *(Mat*)p;
	Mat B;
	if (sigma)
	{
		GaussianBlur(A, B, Size(0, 0), sigma);
		imshow("Sample", B);
	}
	else
		imshow("Sample", A);
}

int main()
{
	namedWindow("Sample");

	Mat A = imread("../fruits.jpg");
	if (A.size().empty())
	{
		cout << "Cannot read image" << endl;
		return 1;
	}

	Vec3b c = A.at<Vec3b>(12, 10);
	cout << "A(12,10)=" << c << endl;
	cout << "mean(A(12,10))=" << (int(c[0]) + int(c[1]) + int(c[2])) / 3 << endl;
	imshow("Sample", A);	waitKey();

	Mat I;
    cvtColor(A, I, COLOR_BGR2GRAY);
    cout << "I(12,10)=" << int(I.at<uchar>(12, 10)) << endl;
	imshow("Sample", I); waitKey();

	int m = I.rows, n = I.cols;
	Mat Ix(m, n, CV_32F), Iy(m, n, CV_32F), G(m, n, CV_32F);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			float ix, iy;
			if (i == 0 || i == m - 1)
				iy = 0;
			else
				iy = (float(I.at<uchar>(i + 1, j)) - float(I.at<uchar>(i - 1, j))) / 2;
			if (j == 0 || j == n - 1)
				ix = 0;
			else
				ix = (float(I.at<uchar>(i, j + 1)) - float(I.at<uchar>(i, j - 1))) / 2;
			Ix.at<float>(i, j) = ix;
			Iy.at<float>(i, j) = iy;
			G.at<float>(i, j) = sqrt(ix*ix + iy * iy);
		}
	}
	cout << "Derivative w.r.t x" << endl;
	imshow("Sample", float2byte(Ix)); waitKey();
	cout << "Derivative w.r.t y" << endl;
	imshow("Sample", float2byte(Iy)); waitKey();
	cout << "Norm of the gradient" << endl;
	imshow("Sample", float2byte(G)); waitKey();

	Mat C;
	threshold(G, C, 10, 1, THRESH_BINARY);
	cout << "Threshold" << endl;
	imshow("Sample", float2byte(C)); waitKey();


	createTrackbar("sigma", "Sample", 0, 20, onTrackbar, &A);
	imshow("Sample", A);	waitKey();

	return 0;
}
