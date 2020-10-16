#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>

#define PI 3.14159265

using namespace cv;
using namespace std;

// Step 1: complete gradient and threshold DONE
// Step 2: complete sobel DONE
// Step 3: complete canny (recommended substep: return Max instead of C to check it) DONE

// Raw gradient. No denoising
void gradient(const Mat&Ic, Mat& G2)
{
	Mat I;
	cvtColor(Ic, I, COLOR_BGR2GRAY);

	int m = I.rows, n = I.cols;
	G2 = Mat(m, n, CV_32F);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			float Ix, Iy;
			// Compute squared gradient (except on borders)
			if (i > 0 && i < m - 1 && j > 0 && j < n - 1) {
				Ix = 0.5 * (-I.at<uchar>(i, j - 1) + I.at<uchar>(i, j + 1));
				Iy = 0.5 * (-I.at<uchar>(i - 1, j) + I.at<uchar>(i + 1, j));
				G2.at<float>(i, j) = sqrt(Ix * Ix + Iy * Iy);
			}
			else G2.at<float>(i, j) = 0.0;
		}
	}
}

// Gradient (and derivatives), Sobel denoising
void sobel(const Mat&Ic, Mat& Ix, Mat& Iy, Mat& G2)
{
	Mat I;
	cvtColor(Ic, I, COLOR_BGR2GRAY);

	int m = I.rows, n = I.cols;
	Ix = Mat(m, n, CV_32F);
	Iy = Mat(m, n, CV_32F);
	G2 = Mat(m, n, CV_32F);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			Ix.at<float>(i, j) = 0.0;
			Iy.at<float>(i, j) = 0.0;
			if (i > 0 && i < m - 1 && j > 0 && j < n - 1) { // Sobel kernel
				Ix.at<float>(i, j) += -I.at<uchar>(i - 1, j - 1) + I.at<uchar>(i - 1, j + 1);
				Ix.at<float>(i, j) += -2 * I.at<uchar>(i, j - 1) + 2 * I.at<uchar>(i, j + 1);
				Ix.at<float>(i, j) += -I.at<uchar>(i + 1, j - 1) + I.at<uchar>(i + 1, j + 1);
				Iy.at<float>(i, j) += -I.at<uchar>(i - 1, j - 1) + -2 * I.at<uchar>(i - 1, j) - I.at<uchar>(i - 1, j + 1);
				Iy.at<float>(i, j) += I.at<uchar>(i + 1, j - 1) + 2 * I.at<uchar>(i + 1, j) + I.at<uchar>(i + 1, j + 1);

				// Normalize magnitude (1/8)
				G2.at<float>(i, j) = 0.125 * sqrt(Ix.at<float>(i, j) * Ix.at<float>(i, j) + Iy.at<float>(i, j) * Iy.at<float>(i, j));
			}
			else G2.at<float>(i, j) = 0.0;
		}
	}
}

// Gradient thresholding, default = do not denoise
Mat threshold(const Mat& Ic, float s, bool denoise = false)
{
	Mat Ix, Iy, G2;
	if (denoise)
		sobel(Ic, Ix, Iy, G2);
	else
		gradient(Ic, G2);
	int m = Ic.rows, n = Ic.cols;
	Mat C(m, n, CV_8U);
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++) {
			if (G2.at<float>(i, j) > s) C.at<uchar>(i, j) = 255; 
			else C.at<uchar>(i, j) = 0;
		}
	return C;
}

// Canny edge detector
Mat canny(const Mat& Ic, float s1)
{
	Mat Ix, Iy, G2;
	sobel(Ic, Ix, Iy, G2);

	float s2 = 2.0 * s1;

	int m = Ic.rows, n = Ic.cols;
	Mat Max(m, n, CV_8U);	// Max pixels ( G2 > s1 && max in the direction of the gradient )
	queue<Point> Q;			// Enqueue seeds ( Max pixels for which G2 > s2 )
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			float l = 255.0, r = 255.0; // store directional pixel values (4 possible directions: 2 axes + 2 diagonals)
			float angle = atan2(Iy.at<float>(i, j), Ix.at<float>(i, j)) * 180 / PI;

			if ((angle > -22.5 && angle <= -22.5) || (angle > 157.5 && angle <= 180) || (angle > -180 && angle <= -157.5) ) {
				l = G2.at<float>(min(i + 1, m), j);
				r = G2.at<float>(max(i - 1, 0), j);
			}
			else if ((angle > 22.5 && angle <= 67.5) || (angle > -157.5 && angle <= -112.5))
			{
				l = G2.at<float>(min(i + 1, m), min(j + 1, n));
				r = G2.at<float>(max(i - 1, 0), max(j - 1, 0));
			}
			else if ((angle > 67.5 && angle <= 112.5) || (angle > -112.5 && angle <= -67.5))
			{
				l = G2.at<float>(i, min(j + 1, n));
				r = G2.at<float>(i, max(j - 1, 0));
			}
			else if ((angle > 112.5 && angle <= 157.5) || (angle > -67.5 && angle <= -22.5) )
			{
				l = G2.at<float>(min(i + 1, m), max(j - 1, 0));
				r = G2.at<float>(max(i - 1, 0), min(j + 1, n));
			}	
			// Recover if max in gradient direction, else set to 0 
			if (G2.at<float>(i, j) > l && G2.at<float>(i, j) > r && G2.at<float>(i, j) > s1)
			{
				//Max.at<uchar>(i, j) = G2.at<float>(i, j);
				Max.at<uchar>(i, j) = 255;
			}
			else Max.at<uchar>(i, j) = 0.0;

			if (G2.at<float>(i, j) > s2) {
				Q.push(Point(j, i));
			}
		}
	}

	// Testing

	imshow("Max", Max);
	waitKey();

	// Propagate seeds
	Mat C(m, n, CV_8U);
	C.setTo(0);
	while (!Q.empty()) {
		int i = Q.front().y, j = Q.front().x;
		Q.pop();
		C.at<uchar>(i, j) = 255;


		if (G2.at<float>(min(i + 1, m), min(j + 1, n)) > s1) C.at<uchar>(min(i + 1, m), min(j + 1, n)) = 255;
		if (Max.at<float>(min(i + 1, m), j            ) > s1) C.at<uchar>(min(i + 1, m), j) = 255;
		if (Max.at<float>(min(i + 1, m), max(j - 1, 0)) > s1) C.at<uchar>(min(i + 1, m), max(j - 1, 0)) = 255;
		if (Max.at<float>(max(i - 1, 0), min(j + 1, n)) > s1) C.at<uchar>(max(i - 1, 0), min(j + 1, n)) = 255;
		if (Max.at<float>(max(i - 1, 0), j            ) > s1) C.at<uchar>(max(i - 1, 0), j) = 255;
		if (Max.at<float>(max(i - 1, 0), max(j - 1, 0)) > s1) C.at<uchar>(max(i - 1, 0), max(j - 1, 0)) = 255;
		if (Max.at<float>(i            , min(j + 1, n)) > s1) C.at<uchar>(i, min(j + 1, n)) = 255;
		if (Max.at<float>(i            , max(j - 1, 0)) > s1) C.at<uchar>(i, max(j - 1, 0)) = 255;

	}

	return C;
}

int main()
{
	Mat I = imread("../road.jpg");

	imshow("Input", I);
	imshow("Threshold", threshold(I, 15));
	imshow("Threshold + denoising", threshold(I, 15, true));
	imshow("Canny", canny(I, 15));

	waitKey();

	return 0;
}
