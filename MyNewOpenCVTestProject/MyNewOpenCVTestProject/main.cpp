#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <vector>
#include <iostream>

using namespace cv;

int main(int argc, char** argv)
{
	// Input Stream from Source
	VideoCapture sequence("C:/training/2/0000000000.png");
	if (!sequence.isOpened())
	{
		std::cerr << "Failed to open Image Sequence!\n" << std::endl;
		return 1;
	}

	Mat frame;	// current frame
	Mat frameEQ;// current frame equalized
	Mat back;	// background image
	Mat fore;	// foreground mask
	Mat mask;	// mask for Noise Cancelation
	Mat test;	// for testing purposes
	vector<Mat> channels;
	
	bool pause = false;
	bool repause = false;

	BackgroundSubtractorMOG2 mog(0, 3, true);
	std::vector<std::vector<Point> > contours;

	while (true)
	{
		if(pause == false){
					sequence >> frame;
		if (frame.empty())
		{
			
			std::cout << "End of Sequence" << std::endl;
			break;
		}

		cvtColor(frame, frameEQ, CV_BGR2YCrCb);
		split(frameEQ, channels);
		equalizeHist(channels[0], channels[0]);
		merge(channels, frameEQ);
		cvtColor(frameEQ, frameEQ, CV_YCrCb2BGR);

		mog.operator()(frameEQ, fore);
		mog.getBackgroundImage(back);

		fore.copyTo(mask);
		erode(mask, mask, Mat::ones(3, 3, CV_8UC1));
		dilate(mask, mask, Mat::ones(20, 20, CV_8UC1));
		threshold(mask, mask, 250, 255, CV_THRESH_BINARY);
		Mat clean = mask.mul(fore);

		GaussianBlur(clean, clean, Size(15, 15), 0, 0);
		threshold(clean, clean, 120, 255, CV_THRESH_BINARY);

		findContours(clean, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		for(int i = contours.size()-1; i > -1; i--){
			if(contours[i].size() < 150)
				contours.erase(contours.begin()+i);
		}
		drawContours(frame, contours, -1, Scalar(200, 255, 0), 1);

		vector<vector<Point> > contours_poly(contours.size());
		vector<Rect> boundRect(contours.size());

		for (int i = 0; i < contours.size(); i++)
		{
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			boundRect[i] = boundingRect(Mat(contours_poly[i]));
			rectangle(frame, boundRect[i].tl(), boundRect[i].br(), 255, 2, 8, 0);
		}

		

		imshow("input | q or esc to quit", frame);
		imshow("background | q or esc to quit", fore);

		if (repause){
			repause = false;
			pause = true;
		}

		}


		char key = (char)waitKey(1);
		if (key == ' ' || key == 'p' || key == 'P'){
			pause = !pause;
		}

		if (key  == '-'){
			repause = true;
			pause = false;
			sequence.set(CV_CAP_PROP_POS_FRAMES, sequence.get(CV_CAP_PROP_POS_FRAMES)-2); //geht 1 bild zurück
		}

		if (key  == '+'){
			repause = true;
			pause = false;
		}


		if (key == 'q' || key == 'Q' || key == 27)
			break;
	}
	
		return 0;
}