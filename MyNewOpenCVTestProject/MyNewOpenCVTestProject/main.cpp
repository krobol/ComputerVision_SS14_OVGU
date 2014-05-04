#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <vector>
#include <iostream>

int main(int argc, char** argv)
{
	// Input Stream from Source
	cv::VideoCapture sequence("D:/Download/training/2/0000000000.png");
	if (!sequence.isOpened())
	{
		std::cerr << "Failed to open Image Sequence!\n" << std::endl;
		return 1;
	}

	cv::Mat frame;	// current frame
	cv::Mat back;	// background image
	cv::Mat fore;	// foreground mask
	cv::Mat structure = cv::Mat::ones(5, 5, CV_8UC1); // strukturelement für dilation
	bool pause = false;
	bool repause = false;
	cv::BackgroundSubtractorMOG2 mog(50, 3, false);
	std::vector<std::vector<cv::Point> > contours;
	//std::cout << "O = " << std::endl << " " << structure << std::endl << std::endl;
	//cv::namedWindow("Image | q or esc to quit", CV_WINDOW_NORMAL);

	while (true)
	{
		if(pause == false){
					sequence >> frame;
		if (frame.empty())
		{
			
			std::cout << "End of Sequence" << std::endl;
			break;
		}

		mog.operator()(frame, fore);
		mog.getBackgroundImage(back);

		cv::erode(fore, fore, cv::Mat::ones(2, 2, CV_8UC1), cv::Point(-1, -1), 1);
		cv::dilate(fore, fore, structure, cv::Point(-1, -1), 1);

	

		cv::findContours(fore, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		for(int i = contours.size()-1; i > -1; i--){
			if(contours[i].size() < 150)
				contours.erase(contours.begin()+i);
		}
		cv::drawContours(frame, contours, -1, cv::Scalar(0, 255, 0), 1);

		
		imshow("input | q or esc to quit", frame);
		//imshow("background | q or esc to quit", fore);

		if (repause){
			repause = false;
			pause = true;
		}

		}


		char key = (char)cv::waitKey(1);
		if (key == 'p' || key == 'P'){
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