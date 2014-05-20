#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <vector>
#include <iostream>
#include <stdlib.h>

using namespace cv;

std::string fileSrc = "C:/training/4/0000000000.png";

Mat& myFilter(Mat& I)
{
	// accept only char type matrices
	CV_Assert(I.depth() != sizeof(uchar));

	int channels = I.channels();

	int nRows = I.rows;
	int nCols = I.cols * channels;

	if (I.isContinuous())
	{
		nCols *= nRows;
		nRows = 1;
	}

	int i, j;
	uchar* p;
	for (i = 0; i < nRows; ++i)
	{
		p = I.ptr<uchar>(i);
		for (j = 0; j < nCols; ++j)
		{
			if (p[j] > 0 && ((p[j - 1] < 255 && p[j + 1] < 255) || (p[j - 1] < 255 && p[j + 2] < 255) || (p[j - 2] < 255 && p[j + 1] < 255) || (p[j - 2] < 255 && p[j + 2] < 255)))
			{
				p[j] = 0;
			}
			if (p[j] == 127) {
				p[j] = 0;
			}
		}
	}

	return I;
}


// prints a string in the upper left corner of the matrix
void printStats(Mat matrix, std::string str) {
	int textWith = str.length() + 1;
	rectangle(matrix, cvPoint(8, 22), cvPoint(12 + 10 * textWith, 8), cvScalar(0, 0, 0), CV_FILLED, 8, 0);
	putText(matrix, str, cvPoint(10, 20), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(255, 255, 255), 1, CV_AA);
}

int main(int argc, char** argv)
{
	// Input Stream from Source
	VideoCapture sequence(fileSrc);
	if (!sequence.isOpened())
	{
		std::cerr << "Failed to open Image Sequence!\n" << std::endl;
		return 1;
	}

	Mat frame;	// current frame
	Mat back;	// background image
	Mat fore;	// foreground mask
	Mat clean;	// clean foreground
	Mat mask;	// mask for cleaner foreground

	vector<Mat> channels;
	int frameCounter = 0;
	bool halt = false;

	bool pause = false;
	bool repause = false;


	BackgroundSubtractorMOG2 mog(0, 3, true);
	std::vector<std::vector<Point> > contours;
	std::vector<std::vector<Point> > oldContours;
	vector<Rect> oldBoundRect;
	vector<int> zuordnung;
	vector<int> zuordnungBackup;

	while (true)
	{
		int start = cvGetTickCount();
		if(pause == false){
			sequence >> frame;
			if (frame.empty())
			{

				std::cout << "End of Sequence" << std::endl;
				break;
			}

			frameCounter++;

			mog.operator()(frame, fore);
			mog.getBackgroundImage(back);

			fore.copyTo(clean);
			clean = myFilter(clean);

			clean.copyTo(mask);
			erode(mask, mask, Mat::ones(3, 3, CV_8UC1));
			dilate(mask, mask, Mat::ones(10, 10, CV_8UC1));

			clean = mask.mul(clean);

			std::vector<std::vector<Point> > contours;
			findContours(clean, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
			for(int i = contours.size()-1; i > -1; i--){
				if(contours[i].size() < 150)
					contours.erase(contours.begin()+i);
			}


			vector<vector<Point> > contours_poly(contours.size());
			vector<Rect> boundRect(contours.size());


			for (unsigned int i = 0; i < contours.size(); i++)
			{
				approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
				boundRect[i] = boundingRect(Mat(contours_poly[i]));
				rectangle(frame, boundRect[i].tl(), boundRect[i].br(), 255, 2, 8, 0);
			}

			/*	vector<int> zuordnung;
			for(unsigned int i = 0; i < oldContours.size(); i++){
			double min = 10000;
			double minIndex = -1;
			for(unsigned int j = 0; j < contours.size(); j++){
			double match = cv::matchShapes(oldContours[i], contours[j], CV_CONTOURS_MATCH_I1, 0);
			if(match < min){
			min = match;
			minIndex = j;
			}
			}
			zuordnung.push_back(minIndex);

			} */

			//surjektive zuordnung alter rechtecke zu neuen rechtecken 

			vector<int> zuordnung;

			for(unsigned int i = 0; i < oldBoundRect.size(); i++){
				if(boundRect.size() == 0) // abbrechen falls keine neuen Rechtecke vorhanden
					break;

				int min = 100000;
				int minIndex = -1;
				for(unsigned int j = 0; j < boundRect.size(); j++){
					int dist = (abs(boundRect[j].tl().x - oldBoundRect[i].tl().x) + abs(boundRect[j].tl().y - oldBoundRect[i].tl().y));
					if(dist < min){
						minIndex = j;
						min = dist;
					}
				}
				zuordnung.push_back(minIndex);
			}


			//mehrere zuordnungen  alter rechtecke zu einem neuen entfernen
			for(unsigned int i = 0; i < zuordnung.size(); i++){
				Vector<int> multipleMapping;
				if(zuordnung[i] == -1)
					continue;
				for(unsigned int j = i; j < zuordnung.size(); j++){
					if(zuordnung[i] == zuordnung[j])
						multipleMapping.push_back(j);
				}
				if(multipleMapping.size() == 1)
					continue;

				int minIndex = -1;
				double min = 10000;
				for(unsigned int j = 0; j < multipleMapping.size(); j++){
					double match = cv::matchShapes(oldContours[multipleMapping[j]], contours[zuordnung[multipleMapping[j]]], CV_CONTOURS_MATCH_I1, 0);
					if(match < min){
						min = match;
						minIndex = j;
					}
				}

				//alle anderen vorerst rauswerfen
				for(unsigned int j = multipleMapping.size()-1; true; j--){
					if(j == minIndex){
						if(j != 0)
							continue;
						else
							break;
					}
					zuordnung[multipleMapping[j]] = -1;
					if(j == 0)
						break;
				}
			}

			//label an neue rechtecke schreiben
			for(unsigned int i = 0; i < zuordnung.size(); i++){
				if(zuordnung[i] < 0)
					continue;
				if(zuordnungBackup.empty())
					putText(frame, std::to_string(i), cvPoint(boundRect[zuordnung[i]].tl().x+50,boundRect[zuordnung[i]].tl().y+50),FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,0,250), 1, CV_AA);
				else{
					for(unsigned int j = 0; j < zuordnungBackup.size(); j++){
						if(zuordnungBackup[j] == i){
							putText(frame, std::to_string(j), cvPoint(boundRect[zuordnung[i]].tl().x+50,boundRect[zuordnung[i]].tl().y+50),FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,0,250), 1, CV_AA);
						}

					}
				}	
			}

			//alte indizes von backup zu neuen machen damit labels gleich bleiben beim nächsten frame
			vector<int> zuordnung2;
			for(unsigned int i = 0; i < zuordnungBackup.size(); i++){
				if(zuordnungBackup[i] == -1){
					zuordnung2.push_back(-1);
					continue;
				}
				if(((int)zuordnung.size())-1 < zuordnungBackup[i])
					continue;
				zuordnung2.push_back(zuordnung[zuordnungBackup[i]]);
			}
			for(unsigned int i = 0; i < zuordnung.size(); i++){
				if(zuordnungBackup.empty())
					break;
				bool found = false;
				for(unsigned int j = 0; j < zuordnung2.size(); j++){
					if(zuordnung2[j] == zuordnung[i]){
						found = true;
						break;
					}
				}
				if(found == false)
					zuordnung2.push_back(zuordnung[i]);
			}

			if(zuordnungBackup.empty())
				zuordnung2 = zuordnung;



			oldBoundRect = boundRect;
			oldContours = contours;
			zuordnungBackup = zuordnung2;

			drawContours(frame, contours, -1, Scalar(200, 255, 0), 1);

			int finish = cvGetTickCount();
			int duration = (int)(cvGetTickFrequency() * 1.0e6) / (finish - start);
			printStats(fore, "#" + std::to_string(frameCounter) + " | " + std::to_string(duration) + " FPS");

			imshow("output | q or esc to quit | spacebar to pause | +/- to go to next/prev frame", frame);
			//imshow("foreground | q or esc to quit | spacebar to pause | +/- to go to next/prev frame", fore);

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