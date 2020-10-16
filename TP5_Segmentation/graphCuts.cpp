#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>

#include "maxflow/graph.h"

#include "image.h"

using namespace std;
using namespace cv;

// This section shows how to use the library to compute a minimum cut on the following graph:
//
//		        SOURCE
//		       /       \
//		     1/         \6
//		     /      4    \
//		   node0 -----> node1
//		     |   <-----   |
//		     |      3     |
//		     \            /
//		     5\          /1
//		       \        /
//		          SINK
//
///////////////////////////////////////////////////

void testGCuts()
{
	Graph<int,int,int> g(/*estimated # of nodes*/ 2, /*estimated # of edges*/ 1); 
	g.add_node(2); 
	g.add_tweights( 0,   /* capacities */  1, 5 );
	g.add_tweights( 1,   /* capacities */  6, 1 );
	g.add_edge( 0, 1,    /* capacities */  4, 3 );
	int flow = g.maxflow();
	cout << "Flow = " << flow << endl;
	for (int i=0;i<2;i++)
		if (g.what_segment(i) == Graph<int,int,int>::SOURCE)
			cout << i << " is in the SOURCE set" << endl;
		else
			cout << i << " is in the SINK set" << endl;
}

Mat float2byte(const Mat& If)
// To imshow matrix and masks 
{
	double minVal, maxVal;
	minMaxLoc(If, &minVal, &maxVal);
	Mat Ib;
	If.convertTo(Ib, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
	return Ib;
}

Mat gradient(Mat& I)
// Computes magnitude of Sobel derivatives from an input color image
{
	// Get grayscale and then float matrix
	Image<uchar> G;
	cvtColor(I, G, COLOR_BGR2GRAY);
	Mat F;
	G.convertTo(F, CV_32F);
	// directional derivatives
	Mat dx, dy;
	Sobel(F, dx, CV_32F, 1, 0);
	Sobel(F, dy, CV_32F, 0, 1);
	// compute sqrt(dx*dx + dy*dy)
	Mat grad;
	grad = dx.mul(dx) + dy.mul(dy);
	sqrt(grad, grad);

	return grad;
}

Mat g(Mat& grad, float& a, float& b)
{
	int n = grad.rows;
	int m = grad.cols;
	Mat result = Mat::zeros(n, m, CV_32F);
	for (unsigned i = 0; i < n; i++)
	{
		for (unsigned j = 0; j < m; j++)
		{
			float partial = 1.0f + b * grad.at<float>(i, j);
			result.at<float>(i, j) = a / partial;
		}
	}
	return result;
}

Mat g_region(Mat& Icolor, Vec3b& I_region)
{
	int n = Icolor.rows;
	int m = Icolor.cols;
	Mat result = Mat::zeros(n, m, CV_32F);
	
	for (unsigned i = 0; i < n; i++)
	{
		for (unsigned j = 0; j < m; j++)
		{
			result.at<float>(i, j) = norm(I_region - Icolor.at<Vec3b>(i, j));
		}
	}

	return result;

}

float lambda(Mat& g, int& n1, int& n2)
{
	// gets 2 node_id, we recover i,j coordinates to look for in g
	int n = g.rows;
	int m = g.cols;

	int j1 = n1 % m;
	int i1 = (n1 - j1) / m;

	int j2 = n2 % m;
	int i2 = (n2 - j2) / m;

	float result = g.at<float>(i1, j1) + g.at<float>(i2, j2);
	return result * 0.5f;
}


int main() {

	// SEGMENTATION BY COLOR SEEDS

	Image<Vec3b> Icolor= Image<Vec3b>(imread("../fishes.jpg"));
	imshow("Source image",Icolor);
	
	//--- Get gradient magnitude (float) to implement g(p)

	Mat grad = gradient(Icolor);
	//imshow("Gradient magnitude", float2byte(grad));

	//--- Compute different g, gi, ge functions

	// alpha, beta and reference color parameters
	float alpha = 1.0f;
	float beta = 1.0f;
	// Vec3b in BGR order: sample colors for background and fish
	Vec3b Ie(0, 175, 150),
		  Ii(255, 200, 200); 

	Mat gMat = g(grad, alpha, beta);
	Mat ge = g_region(Icolor, Ie),
		gi = g_region(Icolor, Ii);

	//imshow("gi", float2byte(gi));
	//imshow("ge", float2byte(ge));
	
	//--- Naive segmentation
	// Make mask by element-wise comparison
	Mat g_diff = (float2byte(gi) > float2byte(ge));

	imshow("G_diff", g_diff);
	

	// SEGMENTATION BY GRAPH CUTS
	
	// Create graph of image

	int n = gMat.rows;
	int m = gMat.cols;

	Graph<int, int, int> G(n * m, 4 * n * m); // I tried with float Graph but nothing happened
	G.add_node(n * m);

	// Add source/target weights and corresponding edges/weights between pixels
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			int node = i * m + j; // Graph class indexing
			int neighbor;         // store neighbor node_id
			int g_e = (int)ge.at<float>(i, j); // for s link
			int g_i = (int)gi.at<float>(i, j); // for t link
			int w_ij; // store lambda_ij 

			G.add_tweights(node, g_e, g_i);

			// as we traverse all nodes in order, we only need edges to lower/right neighbor
			if (i < n - 1) // lower neighbor
			{
				neighbor = (i + 1) * m + j;
				w_ij = (int)lambda(gMat, node, neighbor);
				G.add_edge(node, neighbor, w_ij, w_ij);
			}
			if (j < m - 1) // right neighbor
			{
				neighbor = node + 1;
				w_ij = (int)lambda(gMat, node, neighbor);
				G.add_edge(node, neighbor, w_ij, w_ij);
			}
			/*
			if (i > 0) // upper neighbor
			{
				neighbor = (i - 1) * m + j;
				w_ij = lambda(gMat, node, neighbor);
				G.add_edge(node, neighbor, w_ij, w_ij);
			}
			if (j > 0) // left neighbor
			{
				neighbor = node - 1;
				w_ij = lambda(gMat, node, neighbor);
				G.add_edge(node, neighbor, w_ij, w_ij);
			}
			*/
		}
	}

	int flow = G.maxflow(); // compute max-flow (update node regions)

	cout << flow << endl;

	// recover categories by looking at each node side
	
	Mat GC(n, m, CV_8U);
	
	// Fill matrix accordig to node categories
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			int i_node = i * m + j;
			uchar node_region = (uchar)G.what_segment(i_node);
			GC.at<uchar>(i, j) = 255 * (1 - node_region);
		}
	}

	// Show final result

	imshow("GraphCut", GC);

	waitKey(0);
	return 0;
}
