#include <iostream>
#include <math.h>

#include <opencv2/opencv.hpp>
#include "opencv2/opencv.hpp"
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"


using namespace cv::xfeatures2d;
using namespace std;
using namespace cv;

// Convert to string
#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

RNG rng(12345);

// comparison function object
bool compareContourAreas ( std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 ) {
    double i = fabs( contourArea(cv::Mat(contour1)) );
    double j = fabs( contourArea(cv::Mat(contour2)) );
    return ( i > j );
}


int main()
{

	// Declaration of varaiables
    int count = 0, ind = 0;
    int max_area, cnt_id;
    Rect2d bbox;

    int morph_elem = 1;
	//original - 7
	int morph_size = 1;
	// 0: Opening, 1: Closing
	int morph_operator = 0;
	int operation = morph_operator + 2;

	VideoCapture cap("polwhite1545.mp4");
    
    // Mat frame = imread("np.jpg");

    while (1)
    {

    	Mat frame;
    	cap >> frame;

    	int element_count = 0;
    	Scalar mean_scalar, stddev_scalar;
    	double mean, stddev, Ithresh, nu=0.7;
    	Mat band[3];
    	std::vector<Mat> spec;
    	Mat copy, features, I3c, rgbmin, temp, bgr[3], blue, green, red, ss[3]; //specfree;
    	Mat specfree = Mat::zeros(480, 640, frame.type());
    	Vec3b intensity;
    	copy = frame.clone();
    	features = frame.clone();

    	// imshow("Current", frame);

    	ss[0] = Mat::zeros(480, 640, frame.type());
    	ss[1] = Mat::zeros(480, 640, frame.type());
    	ss[2] = Mat::zeros(480, 640, frame.type());

    	I3c = frame.clone();

        // for (int m = 0; m<frame.rows; m++)
        // {
        //     for (int n = 0; n<frame.cols; n++)
        //     {
        //         intensity = frame.at<Vec3b>(m, n);
        //         rgbmin.at<uchar>(m, n) = min(min(intensity.val[0], intensity.val[1]), intensity.val[2]);
        //     }
        // }

        // cout << "Number of rows: " << rgbmin.rows << ", " << "Number of columns: " << rgbmin.cols << endl;

    	split(I3c, bgr);
    	blue = bgr[0].clone();
    	green = bgr[1].clone();
    	red = bgr[2].clone();

    	cv::min(blue, green, temp);
    	cv::min(temp, red, rgbmin);

    	// for (int m = 0; m<frame.rows; m++)
    	// {
    	// 	for (int n = 0; n<frame.cols; n++)
    	// 	{
    	//        	cout << "X" << (double)rgbmin.at<uchar>(m,n) << endl;
    	// 	}
    	// }

    	cv::meanStdDev(rgbmin, mean_scalar, stddev_scalar);
    	mean = mean_scalar[0];
    	stddev = stddev_scalar[0];

    	// cout << "Mean" << mean << ", " << "Std" << stddev << endl;
    	Ithresh = mean + nu*stddev;
    	//cout << "thresh" << Ithresh << endl;


    	// Approx specular component
    	for (int m=0; m<frame.rows; m++)
    	{
    		for (int n=0; n<frame.cols; n++)
    		{
    			if ((double)rgbmin.at<uchar>(m,n) > Ithresh)
    			{

    				ss[0].at<uchar>(m, n) = (double)blue.at<uchar>(m, n) - (double)rgbmin.at<uchar>(m,n) + Ithresh;
    				ss[1].at<uchar>(m, n) = (double)green.at<uchar>(m, n) - (double)rgbmin.at<uchar>(m,n) + Ithresh;
    				ss[2].at<uchar>(m, n) = (double)red.at<uchar>(m, n) - (double)rgbmin.at<uchar>(m,n) + Ithresh;

    				// ss[0].at<uchar>(m, n) = - (double)rgbmin.at<uchar>(m,n) + Ithresh;
    				// ss[1].at<uchar>(m, n) = - (double)rgbmin.at<uchar>(m,n) + Ithresh;
    				// ss[2].at<uchar>(m, n) = - (double)rgbmin.at<uchar>(m,n) + Ithresh;
    			}
    			else
    			{
    				ss[0].at<uchar>(m, n) = (double)blue.at<uchar>(m, n);
    				ss[1].at<uchar>(m, n) = (double)green.at<uchar>(m, n);
    				ss[2].at<uchar>(m, n) = (double)red.at<uchar>(m, n);

    				// ss[0].at<uchar>(m, n) = (double)0;
    				// ss[1].at<uchar>(m, n) = (double)0;
    				// ss[2].at<uchar>(m, n) = (double)0;
    			}
    		}
    	}


    	spec = {ss[0], ss[1], ss[2]};


    	for (int m = 0; m<frame.rows; m++)
    	{
    		for (int n = 0; n<frame.cols; n++)
    		{
    			specfree.at<Vec3b>(m,n)[0] = ss[0].at<uchar>(m, n);
    			specfree.at<Vec3b>(m,n)[1] = ss[1].at<uchar>(m, n);
    			specfree.at<Vec3b>(m,n)[2] = ss[2].at<uchar>(m, n);

    		}
    	}

    	Mat gray;
        cvtColor(specfree, gray, COLOR_BGR2GRAY);

        // blur(gray, gray, Size(7,7));

        // Original - 50, 255
        // threshold( gray, gray, 10, 50, 0);


        Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
 		// morphologyEx( gray, gray, operation, element );

 		// imshow("Gray", gray);

 		Mat blur_copy, canny;
 		blur(copy, copy, Size(7,7));
        Canny(copy, canny, 100, 255);
        dilate(canny, canny, element);

        imshow("Canny", canny);

        // erode(canny, canny, element);

        

 		vector<vector<Point>> contours, closed_cnts;
        vector<Vec4i> hierarchy;
        findContours(canny, contours, hierarchy, RETR_LIST, CHAIN_APPROX_NONE);

        Mat drawing = Mat::zeros(gray.size(), CV_8UC3);
        for (int i = 0; i < contours.size(); i++)
        {
            Scalar color = Scalar(0,0,255);
            drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
        }


        // sort contours
        std::sort(contours.begin(), contours.end(), compareContourAreas);

        vector<Rect> boundRect(contours.size());
        Scalar color = Scalar(0, 255, 0);

        // Delete all contours with area less than 50
        if (contours.size() > 1) {
            for (int i=0; i<contours.size(); i++) 
            {
                if ((contourArea(contours[i])) < 50 )
                {
                    contours.erase(contours.begin() + i, contours.end());
                }
            }
        }

        
        // Displaying the maximum bounding box
        if (contours.size() > 1) {
            int area_func[contours.size()];
            for (int i=0; i<contours.size(); i++)
            {
                if (contourArea(contours[i])>50 && arcLength(contours[i], true) > 10)
                {
                    area_func[i] = contourArea(contours[i]);
                }
            }
            max_area = *max_element(area_func, area_func+sizeof(area_func)/sizeof(area_func[0]));
            cnt_id = std::distance(area_func,std::max_element(area_func, area_func + sizeof(area_func)/sizeof(area_func[0])));

            bbox = boundingRect(contours[cnt_id]);

            rectangle(frame,bbox, color, 2, 8, 0);
            // output <<  bbox.x << ", " << bbox.y << ", " << bbox.width << ", " << bbox.height << endl;
            // write = true;
        }
        else if (contours.size() == 1){
            for (int i=0; i<contours.size(); i++)
            {
                if (contourArea(contours[0])>50 && arcLength(contours[0], true) > 10)
                {
                    bbox = boundingRect(contours[0]);
                    rectangle(frame, bbox, color, 2, 8, 0);
                    // output <<  bbox.x << ", " << bbox.y << ", " << bbox.width << ", " << bbox.height << endl;
                    // write = true;
                }
                else {
                	// output <<  0 << ", " << 0 << ", " << 0 << ", " << 0 << endl;
                }
            }
        }
        else {
        	// output <<  0 << ", " << 0 << ", " << 0 << ", " << 0 << endl;
        	// write = true;
        }


        //imshow("Canny", drawing);
        // imshow("Contours", drawing);
        imshow("Tracking", frame);
        // imshow("Spec free", specfree);
        waitKey(10);

    	
	}
    
}