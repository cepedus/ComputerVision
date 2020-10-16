#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace std;
using namespace cv;

const float nn_match_ratio = 0.7f; // to distinguish "sufficiently" close keypoints 

int main()
{
	Mat I1 = imread("../IMG_0045.JPG", IMREAD_GRAYSCALE);
	Mat I2 = imread("../IMG_0046.JPG", IMREAD_GRAYSCALE);

	imshow("I1", I1);
	imshow("I2", I2); waitKey();

	/*-------------------------------------------------------------*/

	// Accelerated-KAZE keypoints detector

	// KeyPoint class has size, angle  float attributes, pt point2f point of coordinates
	Ptr<AKAZE> D = AKAZE::create();
	vector<KeyPoint> m1, m2;
	Mat D1, D2, J1, J2;

	// detectAndCompute: input image, input mask, output keypoints, output descriptors
	D->detectAndCompute(I1, noArray(), m1, D1);
	D->detectAndCompute(I2, noArray(), m2, D2);

	//drawKeypoints: img, keypoints, out image, color, flags
	drawKeypoints(I1, m1, J1, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(I2, m2, J2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	// Show new images with keypoints, directions and size
	imshow("I1-KeyPoints", J1);
	imshow("I2-KeyPoints", J2); waitKey();

	/*-------------------------------------------------------------*/

	// Match keypoints by pairs
	Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2);
	vector< vector<DMatch> > nn_matches;
	matcher->knnMatch(D1, D2, nn_matches, 2);
	
	vector<DMatch> matched_keypoints;
	vector<KeyPoint> matched1, matched2;

	// Keep location of good matches
	std::vector<Point2f> points1, points2;

	// Match  sufficiently close keypoints
	for (unsigned i = 0; i < nn_matches.size(); i++) {
		DMatch first = nn_matches[i][0];
		float dist1 = nn_matches[i][0].distance;
		float dist2 = nn_matches[i][1].distance;
		
		if (dist1 < nn_match_ratio * dist2) {
			matched_keypoints.push_back(first);
			matched1.push_back(m1[first.queryIdx]);
			matched2.push_back(m2[first.trainIdx]);
			points1.push_back(m1[first.queryIdx].pt);
			points2.push_back(m2[first.trainIdx].pt);
		}
	}

	// Draw lines between corresponding matches on an unique image
	Mat M_Kpoints;
	drawMatches(I1, m1, I2, m2, matched_keypoints, M_Kpoints);
	imshow("Matched Keypoints", M_Kpoints); waitKey();

	/*---------------------------------------------------------------*/

	// Find homography
	Mat h = findHomography(points2, points1, RANSAC);

	cout << h << endl;

	// Use homography to warp image, show corrected I2
	Mat I_homography;
	Mat Final(I1.rows, I1.cols + I2.cols, CV_8U);

	warpPerspective(I2, I_homography, h, Final.size());

	imshow("Transformed I2", I_homography); waitKey();

	/*---------------------------------------------------------------*/
	// Build final image
	
	for (int i = 0; i < Final.rows; i++)
		for (int j = 0; j < Final.cols; j++) {
			// Override I2 values with warped I2
			if (I_homography.at<uchar>(i, j) > 0) Final.at<uchar>(i, j) = I_homography.at<uchar>(i, j);
			else if (j < I1.cols) Final.at<uchar>(i, j) = I1.at<uchar>(i, j);
		}

	imshow("Merged images", Final);

	waitKey(0);
	return 0;
}
