#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <vector>
#include <iostream>

int main(int argc, char** argv)
{
	// Input Stream from Source
	cv::VideoCapture sequence("C:/training/2/0000000000.png");
	if (!sequence.isOpened())
	{
		std::cerr << "Failed to open Image Sequence!\n" << std::endl;
		return 1;
	}

	cv::Mat frame;	// current frame
	cv::Mat back;	// background image
	cv::Mat fore;	// foreground mask

	cv::BackgroundSubtractorMOG2 mog(0, 16, false);
	std::vector<std::vector<cv::Point> > contours;

	//cv::namedWindow("Image | q or esc to quit", CV_WINDOW_NORMAL);

	for (;;)
	{
		sequence >> frame;
		if (frame.empty())
		{
			std::cout << "End of Sequence" << std::endl;
			break;
		}

		mog.operator()(frame, fore);
		mog.getBackgroundImage(back);

		cv::erode(fore, fore, cv::Mat(), cv::Point(-1, -1), 3);
		cv::dilate(fore, fore, cv::Mat(), cv::Point(-1, -1), 3);

		cv::findContours(fore, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		cv::drawContours(frame, contours, -1, cv::Scalar(0, 255, 0), 1);

		
		imshow("input | q or esc to quit", frame);
		imshow("background | q or esc to quit", back);

		char key = (char)cv::waitKey(1);
		if (key == 'q' || key == 'Q' || key == 27)
			break;
	}

	return 0;
}