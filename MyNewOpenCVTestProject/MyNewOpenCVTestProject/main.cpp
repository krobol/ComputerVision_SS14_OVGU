#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <vector>
#include <iostream>
#include <stdlib.h>

using namespace cv;

std::string fileSrc = "C:/training/7/0000000000.png";

Mat& SaltPepperFilter(Mat& I, int c)
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

	int n, i, j;
	uchar* p;
	for (n = 0; n < c; n++)
	{
		for (i = 0; i < nRows; ++i)
		{
			p = I.ptr<uchar>(i);
			for (j = 0; j < nCols; ++j)
			{
				if (p[j] > 0 && ((p[j - 1] < 255 && p[j + 1] < 255) || (p[j - 2] < 255 && p[j + 2] < 255)))
				{
					p[j] = 0;
				}
				if (p[j] == 127) {
					p[j] = 0;
				}
			}
		}
	}

	return I;
}
void klassifikation(cv::Rect objectToClassify, int dataIndex, vector<vector<vector<int>>> &data)
{
	float fahrzeug = 0.0f;
	float mensch = 0.0f;
	float fahrrad = 0.0f;

	double relation = (double)objectToClassify.width / (double)objectToClassify.height; 
	int maxYChange = data[dataIndex][3][0];
	int yChangeInLast3Frames = 0;
	for(int i = 0; i < data[dataIndex][1].size(); i++)
	{
		yChangeInLast3Frames += abs(data[dataIndex][1][i]);
	}
	if(data[dataIndex][1].size() < 3)
		yChangeInLast3Frames = 0;
	/*
	if(maxYChange == 0)
	{
		mensch += 3.0f/367.0f;
		fahrzeug += 5.0f/529.0f;
	}
	if(maxYChange == 1)
	{
		mensch += 8.0f/367.0f;
		fahrzeug += 17.0f/529.0f;
	}
	if(maxYChange == 2)
	{
		fahrrad += 2.0f/82.0f;
		mensch += 24.0f/367.0f;
		fahrzeug += 66.0f/529.0f;
	}
	if(maxYChange == 3)
	{
		fahrrad += 1.0f/82.0f;
		mensch += 25.0f/367.0f;
		fahrzeug += 43.0f/529.0f;
	}
	if(maxYChange == 4)
	{
		fahrrad += 8.0f/82.0f;
		mensch += 39.0f/367.0f;
		fahrzeug += 42.0f/529.0f;
	}
	if(maxYChange == 5)
	{
		mensch += 5.0f/367.0f;
		fahrzeug += 55.0f/529.0f;
	}
	if(maxYChange == 6)
	{
		mensch += 15.0f/367.0f;
		fahrzeug += 41.0f/529.0f;
	}
	if(maxYChange == 7)
	{
		mensch += 15.0f/367.0f;
		fahrzeug += 18.0f/529.0f;
	}
	if(maxYChange == 8)
	{
		mensch += 3.0f/367.0f;
		fahrzeug += 22.0f/529.0f;
	}
	if(maxYChange == 9)
	{
		fahrrad += 6.0f/82.0f;
		mensch += 2.0f/367.0f;
	}
	if(maxYChange == 10)
	{
		fahrrad += 1.0f/82.0f;
		mensch += 16.0f/367.0f;
	}
	if(maxYChange == 11)
	{
		fahrrad += 8.0f/82.0f;
		mensch += 15.0f/367.0f;
		fahrzeug += 12.0f/529.0f;
	}
	if(maxYChange == 12)
	{
		fahrrad += 17.0f/82.0f;
		mensch += 25.0f/367.0f;
		fahrzeug += 7.0f/529.0f;
	}
	if(maxYChange == 13)
	{
		fahrrad += 6.0f/82.0f;
	}
	if(maxYChange == 14)
	{
		fahrzeug += 3.0f/529.0f;
	}
	if(maxYChange == 15)
	{
		fahrzeug += 2.0f/529.0f;
	}
	if(maxYChange == 16)
	{
		mensch += 5.0f/367.0f;
		fahrzeug += 51.0f/529.0f;
	}
	if(maxYChange == 17)
	{
		fahrzeug += 37.0f/529.0f;
	}
	if(maxYChange == 18)
	{
		fahrzeug += 1.0f/529.0f;
	}
	if(maxYChange == 19)
	{
		mensch += 1.0f/367.0f;
	}
	if(maxYChange >= 20 && maxYChange < 25)
	{
		fahrrad += 1.0f/82.0f;
		mensch += 14.0f/367.0f;
		fahrzeug += 22.0f/529.0f;
	}
	if(maxYChange >= 25 && maxYChange < 75)
	{
		fahrrad += 13.0f/82.0f;
		mensch += 5.0f/367.0f;
		fahrzeug += 23.0f/529.0f;
	}
	if(maxYChange >= 75 && maxYChange < 100)
	{
		fahrrad += 8.0f/82.0f;
		fahrzeug += 6.0f/529.0f;
	}
	if(maxYChange >= 100)
	{
		fahrrad += 8.0f/82.0f;
		mensch += 163.0f/367.0f;
		fahrzeug += 62.0f/529.0f;
	}*/

	if(yChangeInLast3Frames == 1)
	{
		fahrrad += 3.0f/148.0f;
		mensch += 4.0f/222.0f;
		fahrzeug += 42.0f/501.0f;
	}
	if(yChangeInLast3Frames == 2)
	{
		fahrrad += 6.0f/148.0f;
		mensch += 8.0f/222.0f;
		fahrzeug += 65.0f/501.0f;
	}
	if(yChangeInLast3Frames == 3)
	{
		fahrrad += 11.0f/148.0f;
		mensch += 9.0f/222.0f;
		fahrzeug += 73.0f/501.0f;
	}
	if(yChangeInLast3Frames == 4)
	{
		fahrrad += 20.0f/148.0f;
		mensch += 26.0f/222.0f;
		fahrzeug += 50.0f/501.0f;
	}
	if(yChangeInLast3Frames == 5)
	{
		fahrrad += 20.0f/148.0f;
		mensch += 16.0f/222.0f;
		fahrzeug += 54.0f/501.0f;
	}
	if(yChangeInLast3Frames == 6)
	{
		fahrrad += 14.0f/148.0f;
		mensch += 28.0f/222.0f;
		fahrzeug += 31.0f/501.0f;
	}
	if(yChangeInLast3Frames == 7)
	{
		fahrrad += 11.0f/148.0f;
		mensch += 18.0f/222.0f;
		fahrzeug += 33.0f/501.0f;
	}
	if(yChangeInLast3Frames == 8)
	{
		fahrrad += 11.0f/148.0f;
		mensch += 15.0f/222.0f;
		fahrzeug += 23.0f/501.0f;
	}
	if(yChangeInLast3Frames == 9)
	{
		fahrrad += 6.0f/148.0f;
		mensch += 5.0f/222.0f;
		fahrzeug += 15.0f/501.0f;
	}
	if(yChangeInLast3Frames == 10)
	{
		fahrrad += 6.0f/148.0f;
		mensch += 13.0f/222.0f;
		fahrzeug += 10.0f/501.0f;
	}
	if(yChangeInLast3Frames == 11)
	{
		fahrrad += 4.0f/148.0f;
		mensch += 3.0f/222.0f;
		fahrzeug += 9.0f/501.0f;
	}
	if(yChangeInLast3Frames == 12)
	{
		fahrrad += 4.0f/148.0f;
		mensch += 10.0f/222.0f;
		fahrzeug += 5.0f/501.0f;
	}
	if(yChangeInLast3Frames == 13)
	{
		fahrrad += 1.0f/148.0f;
		mensch += 9.0f/222.0f;
		fahrzeug += 8.0f/501.0f;
	}
	if(yChangeInLast3Frames == 14)
	{
		fahrrad += 3.0f/148.0f;
		mensch += 1.0f/222.0f;
		fahrzeug += 3.0f/501.0f;
	}
	if(yChangeInLast3Frames == 15)
	{
		fahrrad += 4.0f/148.0f;
		mensch += 4.0f/222.0f;
		fahrzeug += 5.0f/501.0f;
	}
	if(yChangeInLast3Frames == 16)
	{
		fahrrad += 2.0f/148.0f;
		mensch += 5.0f/222.0f;
		fahrzeug += 5.0f/501.0f;
	}
	if(yChangeInLast3Frames == 17)
	{
		fahrrad += 1.0f/148.0f;
		mensch += 1.0f/222.0f;
		fahrzeug += 2.0f/501.0f;
	}
	if(yChangeInLast3Frames == 18)
	{
		mensch += 5.0f/222.0f;
		fahrzeug += 5.0f/501.0f;
	}
	if(yChangeInLast3Frames == 19)
	{
		mensch += 3.0f/222.0f;
		fahrzeug += 2.0f/501.0f;
	}
	if(yChangeInLast3Frames == 20)
	{
		mensch += 1.0f/222.0f;
		fahrzeug += 2.0f/501.0f;
	}
	if(yChangeInLast3Frames > 20 && yChangeInLast3Frames <= 25)
	{
		fahrrad += 6.0f/148.0f;
		mensch += 5.0f/222.0f;
		fahrzeug += 20.0f/501.0f;
	}
	if(yChangeInLast3Frames > 25 && yChangeInLast3Frames <= 75)
	{
		fahrrad += 5.0f/148.0f;
		mensch += 2.0f/222.0f;
		fahrzeug += 35.0f/501.0f;
	}
	if(yChangeInLast3Frames > 100)
	{
		fahrrad += 10.0f/148.0f;
		mensch += 29.0f/222.0f;
		fahrzeug += 8.0f/501.0f;
	}

	if(relation >= 0.1f && relation < 0.2f)
	{
		fahrrad += 1.0f/82.0f;
	}
	if(relation >= 0.2f && relation < 0.3f)
	{
		fahrrad += 3.0f/82.0f;
		mensch += 9.0f/367.0f;
	}
	if(relation >= 0.3f && relation < 0.4f)
	{
		mensch += 39.0f/367.0f;
		fahrzeug += 1.0f/535.0f;
	}
	if(relation >= 0.4f && relation < 0.5f)
	{
		fahrrad += 9.0f/82.0f;
		mensch += 49.0f/367.0f;
		fahrzeug += 1.0f/535.0f;
	}
	if(relation >= 0.5f && relation < 0.6f)
	{
		fahrrad += 10.0f/82.0f;
		mensch += 32.0f/367.0f;
		fahrzeug += 4.0f/535.0f;
	}
	if(relation >= 0.6f && relation < 0.7f)
	{
		fahrrad += 9.0f/82.0f;
		mensch += 33.0f/367.0f;
		fahrzeug += 5.0f/535.0f;
	}
	if(relation >= 0.7f && relation < 0.8f)
	{
		fahrrad += 7.0f/82.0f;
		mensch += 34.0f/367.0f;
		fahrzeug += 3.0f/535.0f;
	}
	if(relation >= 0.8f && relation < 0.9f)
	{
		fahrrad += 10.0f/82.0f;
		mensch += 24.0f/367.0f;
		fahrzeug += 10.0f/535.0f;
	}
	if(relation >= 0.9f && relation < 1.0f)
	{
		fahrrad += 18.0f/82.0f;
		mensch += 20.0f/367.0f;
		fahrzeug += 11.0f/535.0f;
	}
	if(relation >= 1.0f && relation < 1.1f)
	{
		fahrrad += 5.0f/82.0f;
		mensch += 23.0f/367.0f;
		fahrzeug += 8.0f/535.0f;
	}
	if(relation >= 1.1f && relation < 1.2f)
	{
		fahrrad += 3.0f/82.0f;
		mensch += 17.0f/367.0f;
		fahrzeug += 7.0f/535.0f;
	}
	if(relation >= 1.2f && relation < 1.3f)
	{
		fahrrad += 3.0f/82.0f;
		mensch += 15.0f/367.0f;
		fahrzeug += 19.0f/535.0f;
	}
	if(relation >= 1.3f && relation < 1.4f)
	{
		fahrrad += 1.0f/82.0f;
		mensch += 23.0f/367.0f;
		fahrzeug += 21.0f/535.0f;
	}
	if(relation >= 1.4f && relation < 1.5f)
	{
		fahrrad += 1.0f/82.0f;
		mensch += 15.0f/367.0f;
		fahrzeug += 21.0f/535.0f;
	}
	if(relation >= 1.5f && relation < 1.6f)
	{
		mensch += 6.0f/367.0f;
		fahrzeug += 18.0f/535.0f;
	}
	if(relation >= 1.6f && relation < 1.7f)
	{
		mensch += 8.0f/367.0f;
		fahrzeug += 21.0f/535.0f;
	}
	if(relation >= 1.7f && relation < 1.8f)
	{
		mensch += 2.0f/367.0f;
		fahrzeug += 24.0f/535.0f;
	}
	if(relation >= 1.8f && relation < 1.9f)
	{
		mensch += 1.0f/367.0f;
		fahrzeug += 33.0f/535.0f;
	}
	if(relation >= 1.9f && relation < 2.0f)
	{
		fahrzeug += 31.0f/535.0f;
	}
	if(relation >= 2)
	{
		fahrrad += 2.0f/82.0f;
		mensch += 19.0f/367.0f;
		fahrzeug += 297.0f/535.0f;
	}

	if(data[dataIndex][6][0] == 0){
		fahrzeug += 0.5;
	}
	if(data[dataIndex][6][0] == 1){
		mensch += 0.5;
	}
	if(data[dataIndex][6][0] == 2){
		fahrrad += 0.5;
	}

	if(fahrzeug >= mensch && fahrzeug >= fahrrad)
		data[dataIndex][6][0] = 0;
	else if (mensch >= fahrzeug && mensch >= fahrrad)
		data[dataIndex][6][0] = 1;
	else if (fahrrad >= fahrzeug && fahrrad >= mensch)
		data[dataIndex][6][0] = 2;
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

	Mat frame;			// current frame
	Mat uniform;		// frame with uniform intensity
	Mat back;			// background image
	Mat fore;			// foreground mask
	Mat mask;			// mask for cleaner foreground
	Mat denoised;		// foreground mask with less noise		
	Mat prev;			// previous frame's denoised foreground mask
	Mat deartifacted;	// foreground mask with less lighting artifacts
	Mat result;			// result of all image processing
	
	Mat phantom;
	Mat trails;
	Mat check;

	int cAll = 0;
	int c0 = 0;
	int c1 = 0;
	int c2 = 0;
	int c3 = 0;
	int c4 = 0;
	int c5 = 0;
	int c6 = 0;
	int c7 = 0;
	int c8 = 0;
	int c9 = 0;
	int c10 = 0;
	int c11 = 0;
	int c12 = 0;
	int c13 = 0;
	int c14 = 0;
	int c15 = 0;
	int c16 = 0;
	int c17 = 0;
	int c18 = 0;
	int c19 = 0;
	int c20 = 0;
	int c21 = 0;
	int c22 = 0;
	int c23 = 0;
	int c24 = 0;
	int c25 = 0;
	int c26 = 0;
	int c27 = 0;
	int c28 = 0;

	vector<Mat> channels;
	vector<Mat> bg_channels;

	bool halt = false;

	bool pause = false;
	bool repause = false;


	BackgroundSubtractorMOG2 mog(0, 3, true);
	std::vector<std::vector<Point> > contours;
	std::vector<std::vector<Point> > oldContours;
	vector<Rect> oldBoundRect;
	vector<int> zuordnung;
	vector<int> zuordnungBackup;
	vector<vector<vector<int>>> data; // geschwindigkeit x, geschwindigkeiten y, durchschnittliche ver�nderung der geschwindigkeit in x richtung, durchschnittliche ver�nderung der geschwindigkeit in y richtung, frame in dem objekt gefunden, fl�che zu beginn, klassifizierung 0 = auto, 1 = mensch, 2 = fahrrad

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

			//std::cout << sequence.get(CV_CAP_PROP_POS_FRAMES) << std::endl;

			phantom = Mat::zeros(frame.size[0], frame.size[1], CV_32FC3);
			if (trails.empty()) trails = Mat::zeros(frame.size[0], frame.size[1], CV_32FC3);


			///// Background Subtraction /////

			cvtColor(frame, uniform, CV_BGR2HSV);
			split(uniform, channels);

			channels[2] = Mat::zeros(frame.size[0], frame.size[1], CV_8UC1);        // Helligkeit = 0

			merge(channels, uniform);

			mog.operator()(uniform, fore);
			mog.getBackgroundImage(back);

			///// Image Processing /////

			fore.copyTo(mask);

			SaltPepperFilter(mask, 3);
			erode(mask, mask, Mat::ones(3, 6, CV_8UC1));
			dilate(mask, mask, Mat::ones(15, 15, CV_8UC1));

			denoised = fore & mask;
			GaussianBlur(denoised, denoised, Size(9, 9), 0, 0);
			threshold(denoised, denoised, 200, 255, THRESH_BINARY);

			if (!prev.empty())
			{
				deartifacted = prev & denoised;
			}
			else
			{
				deartifacted = Mat::ones(frame.size[0], frame.size[1], CV_8UC1);
			}

			denoised.copyTo(prev);
			dilate(deartifacted, deartifacted, Mat::ones(20, 120, CV_8UC1));

			result = denoised & deartifacted;

			///// Connected Component Analysis /////

			std::vector<std::vector<Point> > contours;
			findContours(result, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

			for(int i = contours.size()-1; i > -1; i--){
				if(contours[i].size() < 100)
					contours.erase(contours.begin()+i);
			}

			vector<vector<Point> > contours_poly(contours.size());
			vector<Rect> boundRect(contours.size());


			for (unsigned int i = 0; i < contours.size(); i++)
			{
				approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
				boundRect[i] = boundingRect(Mat(contours_poly[i]));

				// erase all objects that are thinner than a certain threshold
				// TODO: Exclude cases where Rectangle is thin because of cut off at the frame's edges
				if (boundRect[i].height / boundRect[i].width > 5)
				{
					contours.erase(contours.begin() + i);
					boundRect.erase(boundRect.begin() + i);
				}

				//rectangle(frame, boundRect[i].tl(), boundRect[i].br(), 255, 2, 8, 0);

			}
			//std::cout << sequence.get(CV_CAP_PROP_POS_FRAMES) << std::endl;

			//surjektive zuordnung alter rechtecke zu neuen rechtecken 

			vector<int> zuordnung;
			for(unsigned int i = 0; i < oldBoundRect.size(); i++){
				if(boundRect.size() == 0) // abbrechen falls keine neuen Rechtecke vorhanden
					break;

				int min = 100000;
				int minIndex = -1;
				for(unsigned int j = 0; j < boundRect.size(); j++){
					int dist = (abs(boundRect[j].tl().x - oldBoundRect[i].tl().x) + abs(boundRect[j].tl().y - oldBoundRect[i].tl().y));

					//richtung pruefen
					if(!zuordnungBackup.empty()){
						int index = -1;
						for(unsigned int k = 0; k < zuordnungBackup.size(); k++){
							if(zuordnungBackup[k] == i){
								index = k;
								break;
							}
						}
						if(index >= 0){
							int currentRichtung = boundRect[j].tl().x - oldBoundRect[i].tl().x;
							if(currentRichtung > 0 && data[index][0][0] < 0 || currentRichtung < 0 && data[index][0][0] > 0)
								continue;
						}
					}

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


				//konturen miteinander vergleichen und aehnlichste kontur finden
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

			//verdeckungen oder besser gesagt zertrennung
			for(unsigned int i = 0; i < zuordnung.size(); i++){
				//break;
				if(zuordnungBackup.empty())
					break;
				if(zuordnung[i] < 0){
					continue;
				}

				int index = -1;
				for(int j = 0; j < zuordnungBackup.size(); j++){
					if(zuordnungBackup[j] == i){
						index = j;
						break;
					}
				}
				if(index == -1)
					continue;

				if(data[index][5][0] == -1)
					continue;
				if(boundRect[zuordnung[i]].width * boundRect[zuordnung[i]].height >= (float)data[index][5][0])
					continue;

				vector<int> nichtZugeordnet(boundRect.size());
				for(unsigned int j = 0; j < zuordnung.size(); j++){
					if(zuordnung[j] == -1){
						continue;
					}
					nichtZugeordnet[zuordnung[j]] = 1;
				}


				vector<int> possibleCandidates; // wird alle rechtecke enthalten die nur maximal 50pixel entfernt sind
				for(unsigned int j = 0; j < nichtZugeordnet.size(); j++){
					if(nichtZugeordnet[j] == 1) // bereits zugeordnete elemente ignorieren
						continue;
					if(abs(boundRect[j].br().x - boundRect[zuordnung[i]].br().x) <= 50 || abs(boundRect[j].br().x - boundRect[zuordnung[i]].tl().x) || abs(boundRect[j].tl().x - boundRect[zuordnung[i]].br().x) || abs(boundRect[j].tl().x - boundRect[zuordnung[i]].tl().x))
						possibleCandidates.push_back(j);
				}

				if(possibleCandidates.size() == 0)
					continue;

				int minX = 10000;
				int maxX = -10000;
				int minY = 10000;
				int maxY = -10000;
				int minIndex = -1;
				int minDiff = 100000000;
				for(int j = 0; j < possibleCandidates.size(); j++){

					//rechtecke vereinen
					if(boundRect[zuordnung[i]].tl().x < minX)
						minX = boundRect[zuordnung[i]].tl().x;
					if(boundRect[zuordnung[i]].br().x < minX)
						minX = boundRect[zuordnung[i]].br().x;
					if(boundRect[possibleCandidates[j]].tl().x < minX)
						minX = boundRect[possibleCandidates[j]].tl().x;
					if(boundRect[possibleCandidates[j]].br().x < minX)
						minX = boundRect[possibleCandidates[j]].br().x;


					if(boundRect[zuordnung[i]].tl().x > maxX)
						maxX = boundRect[zuordnung[i]].tl().x;
					if(boundRect[zuordnung[i]].br().x > maxX)
						maxX = boundRect[zuordnung[i]].br().x;
					if(boundRect[possibleCandidates[j]].tl().x > maxX)
						maxX = boundRect[possibleCandidates[j]].tl().x;
					if(boundRect[possibleCandidates[j]].br().x > maxX)
						maxX = boundRect[possibleCandidates[j]].br().x;


					if(boundRect[zuordnung[i]].tl().y < minY)
						minY = boundRect[zuordnung[i]].tl().y;
					if(boundRect[zuordnung[i]].br().y < minY)
						minY = boundRect[zuordnung[i]].br().y;
					if(boundRect[possibleCandidates[j]].tl().y < minY)
						minY = boundRect[possibleCandidates[j]].tl().y;
					if(boundRect[possibleCandidates[j]].br().y < minY)
						minY = boundRect[possibleCandidates[j]].br().y;


					if(boundRect[zuordnung[i]].tl().y > maxY)
						maxY = boundRect[zuordnung[i]].tl().y;
					if(boundRect[zuordnung[i]].br().y > maxY)
						maxY = boundRect[zuordnung[i]].br().y;
					if(boundRect[possibleCandidates[j]].tl().y > maxY)
						maxY = boundRect[possibleCandidates[j]].tl().y;
					if(boundRect[possibleCandidates[j]].br().y > maxY)
						maxY = boundRect[possibleCandidates[j]].br().y;

					int flaeche = (maxX - minX) * (maxY - minY);
					if(flaeche > data[index][5][0]*1.5)
						continue;



					//if(abs(flaeche - data[index][5]) < abs(data[index][5] - boundRect[zuordnung[i]].width * boundRect[zuordnung[i]].height)){
					boundRect[zuordnung[i]] = cv::Rect(minX, minY, maxX - minX, maxY - minY); //neues vereintes rechteck erstellen
					boundRect[possibleCandidates[j]].x = -10000;
					boundRect[possibleCandidates[j]].y = -10000;
					//}


				}

			}

			//label an neue rechtecke schreiben
			for(unsigned int i = 0; i < zuordnung.size(); i++){
				if(zuordnung[i] < 0){
					continue;
				}

				cv::Scalar color = cv::Scalar(0,0,0);


				if(zuordnungBackup.empty()){
					putText(frame, std::to_string(i), cvPoint(boundRect[zuordnung[i]].tl().x+50,boundRect[zuordnung[i]].tl().y+50),FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,0,250), 1, CV_AA);
					//rectangle(frame, boundRect[zuordnung[i]].tl(), boundRect[zuordnung[i]].br(), color, 2, 8, 0);

					std::vector<std::vector<Point> > singleContour;
					singleContour.push_back(contours[zuordnung[i]]);
					drawContours(frame, singleContour, -1, Scalar(200, 255, 0), 1);
				}               
				else{
					for(unsigned int j = 0; j < zuordnungBackup.size(); j++){
						if(zuordnungBackup[j] == i){
							
							if(data[j][6][0] == 0)
								color = cv::Scalar(0,0,255);
							if(data[j][6][0] == 1)
								color = cv::Scalar(255,0,0);
							if(data[j][6][0] == 2)
								color = cv::Scalar(0,255,0);

							std::vector<std::vector<Point> > singleContour;
							singleContour.push_back(contours[zuordnung[i]]);
							
							Vec3b pixel = check.at<Vec3b>(boundRect[zuordnung[i]].y + (boundRect[zuordnung[i]].height / 2), boundRect[zuordnung[i]].x + (boundRect[zuordnung[i]].width / 2));

							int c = 0;
							if (pixel[1] + pixel[2] + pixel[3] == 0) c = 7;
							if (pixel[1] != 0 || pixel[2] != 0 || pixel[3] != 0) c = 5;
							if (pixel[1] > 0 && pixel[2] > 0 && pixel[3] > 0) c = 2;
							if (sequence.get(CV_CAP_PROP_POS_FRAMES) < 50) c = 1;


							if(sequence.get(CV_CAP_PROP_POS_FRAMES) - data[j][4][0] >= c)
							{
								drawContours(frame, singleContour, -1, Scalar(200, 255, 0), 1);
								drawContours(phantom, singleContour, -1, color, CV_FILLED);
								rectangle(frame, boundRect[zuordnung[i]].tl(), boundRect[zuordnung[i]].br(), color, 2, 8, 0);
								putText(frame, std::to_string(j), cvPoint(boundRect[zuordnung[i]].tl().x+50,boundRect[zuordnung[i]].tl().y+50),FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,0,250), 1, CV_AA);

							}
							
						}

					}
				}       
			}

			//alte indizes von backup zu neuen machen damit labels gleich bleiben beim naechsten frame
			vector<int> zuordnung2;
			for(unsigned int i = 0; i < zuordnungBackup.size(); i++){
				if(zuordnungBackup[i] == -1){
					zuordnung2.push_back(-1);

					continue;
				}
				if(((int)zuordnung.size())-1 < zuordnungBackup[i]){
					zuordnung2.push_back(-1);

					continue;
				}

				zuordnung2.push_back(zuordnung[zuordnungBackup[i]]);



			}



			//neue objekte zu zuordnung2 hinzufuegen
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
				if(found == false){
					zuordnung2.push_back(zuordnung[i]);
				}
			}

			if(zuordnungBackup.empty())
				zuordnung2 = zuordnung;

			zuordnungBackup = zuordnung2;


			//data erstellen bzw. updaten
			for(int i = 0; i < zuordnungBackup.size(); i++){

				//falls neues objekt
				if(i >= data.size()){

					//falls verlorenes object
					if(zuordnungBackup[i] == -1){
						vector<vector<int>> newData;
						vector<int> null;
						null.push_back(0);
						vector<int> minusOne;
						minusOne.push_back(-1);
						newData.push_back(null); // x richtung
						newData.push_back(null); // y richtung
						newData.push_back(null); // durchschnitt veraenderung x
						newData.push_back(null); // durchschnitt veraenderung y
						newData.push_back(null); // frame 
						newData.push_back(null); // flaeche
						newData.push_back(minusOne); // klassifizierung
						data.push_back(newData);
						continue;
					}

					vector<vector<int>> newData;
					int index = -1;
					for(unsigned int j = 0; j < zuordnung.size(); j++){
						if(zuordnung[j] == zuordnungBackup[i]){
							index = j;
							break;
						}
					}
					vector<int> xChange;
					vector<int> yChange;
					vector<int> averageXChange;
					vector<int> maxYChange;
					vector<int> frame;
					vector<int> flaeche;
					vector<int> klasse;

					xChange.push_back(boundRect[zuordnungBackup[i]].tl().x - oldBoundRect[index].tl().x);
					yChange.push_back(boundRect[zuordnungBackup[i]].tl().y - oldBoundRect[index].tl().y);
					averageXChange.push_back(0);
					maxYChange.push_back(boundRect[zuordnungBackup[i]].tl().y - oldBoundRect[index].tl().y);
					frame.push_back(sequence.get(CV_CAP_PROP_POS_FRAMES));

					newData.push_back(xChange); // x richtung
					newData.push_back(yChange); // y richtung
					newData.push_back(averageXChange); // durchschnitt veraenderung x
					newData.push_back(maxYChange); // groesste veraenderung y
					newData.push_back(frame); // frame


					if(boundRect[zuordnungBackup[i]].br().x >= sequence.get(CV_CAP_PROP_FRAME_WIDTH)-1 || boundRect[zuordnungBackup[i]].tl().x <= 1)
					{
						flaeche.push_back(-1);
						newData.push_back(flaeche); // objekt ausserhalb des bildes

					}
					else
					{
						flaeche.push_back(oldBoundRect[index].area());
						newData.push_back(flaeche); // flaeche
					}

					klasse.push_back(-1);
					newData.push_back(klasse); // klassifizierung

					data.push_back(newData);
					continue;
				}

				//ansonsten updaten
				if(zuordnungBackup[i] == -1)
					continue;

				int index = -1;
				for(int j = 0; j < zuordnung.size(); j++){
					if(zuordnung[j] == zuordnungBackup[i]){
						index = j;
						break;
					}
				}

				int newRichtungX = boundRect[zuordnungBackup[i]].tl().x - oldBoundRect[index].tl().x;
				int newRichtungY = abs(boundRect[zuordnungBackup[i]].tl().y - oldBoundRect[index].tl().y);
				/*int alterDurchschnittX = data[i][2][0];
				int neuerDurchschnittX = (((alterDurchschnittX * ((sequence.get(CV_CAP_PROP_POS_FRAMES)-1) - data[i][4][0])) + (newRichtungX - data[i][0][0])) / (sequence.get(CV_CAP_PROP_POS_FRAMES) - data[i][4][0]));
				int neuerDurchschnittX2 = (((alterDurchschnittX * ((sequence.get(CV_CAP_PROP_POS_FRAMES)-1) - data[i][4][0])) + (abs(abs(newRichtungX) - data[i][0][0]))) / (sequence.get(CV_CAP_PROP_POS_FRAMES) - data[i][4][0]));
				int alterDurchschnittY = data[i][3][0];
				int neuerDurchschnittY = (((alterDurchschnittY * ((sequence.get(CV_CAP_PROP_POS_FRAMES)-1) - data[i][4][0])) + (abs(newRichtungY - data[i][1][0]))) / (sequence.get(CV_CAP_PROP_POS_FRAMES) - data[i][4][0]));*/
				data[i][0][0] = newRichtungX;
				//data[i][1][0] = newRichtungY;
				if(data[i][1].size() < 3)
					data[i][1].push_back(newRichtungY);
				else
				{
					data[i][1].erase(data[i][1].begin());
					data[i][1].push_back(newRichtungY);
				}
				if(newRichtungY > data[i][3][0])
					data[i][3][0] = newRichtungY;

				if(data[i][5].size() < 3)
					data[i][5].push_back(boundRect[zuordnungBackup[i]].area());
				else
				{
					data[i][5].erase(data[i][5].begin());
					data[i][5].push_back(boundRect[zuordnungBackup[i]].area());
				}

				//data[i][2][0] = neuerDurchschnittX;
				//data[i][3][0] = neuerDurchschnittY;
				if(data[i][5].size() == 3)
				{
				//if(i == 49 || i == 84){ //fahrrad sequenz 7
				if(i == 138 || i == 165 || i == 194 || i == 178 || i == 209 || i == 181){ //fahrrad sequenz 6
				//if(i == 47 || i == 50 || i == 52 || i == 77 || i == 97 || i == 136 || i == 154 || i == 149 || i == 166 || i == 176){ //menschen sequenz 6
				//if(i == 1 || i == 3){ //menschen sequenz 5	
				//if(i == 13 || i == 15 || i == 26 || i == 38 || i == 44 || i == 101 || i == 109 || i == 121 || i == 120  || i == 131 || i == 141 || i == 147 || i == 150 || i == 143 || i == 209 || i == 382 || i == 427){ //menschen sequezn 4
				//if(i == 17 || i == 12 || i == 33 || i == 37){//menschen sequenz 3
				//if(i == 3 || i == 4 || i == 5 || i == 7 || i == 8){//autos sequenz 1
				//if(i == 0 || i == 3 || i == 4 || i == 5 || i == 18 || i == 17 || i == 19 || i == 22 || i == 24 || i == 29 || i == 35 || i == 34 || i == 43 || i == 53){//auto sequenz 2
				//if(i == 11 || i == 16 || i == 46 || i == 60 || i == 80 || i == 108 || i == 188 || i == 195 || i == 201 || i == 213 || i == 225 || i == 218){ // autos sequenz 3
				//if(i == 23 || i == 29|| i == 48){//autos sequenz 6

					int yChangeInLast3Frames = 0;
					int counter = 0;
					for(int j = 0; j < data[i][5].size(); j++)
					{
						if(data[i][5][j] == -1 || j == 0)
							continue;
						if(data[i][5][j-1] == -1)
							continue;
							yChangeInLast3Frames += abs((data[i][5][j] - data[i][5][j-1]));
							counter++;
					}
					yChangeInLast3Frames /= counter;

					

					cAll++;
					if(yChangeInLast3Frames < 100)
						c0++;
					if(yChangeInLast3Frames >= 100 && yChangeInLast3Frames < 500)
						c1++;
					if(yChangeInLast3Frames >= 500 && yChangeInLast3Frames < 1000)
						c2++;
					if(yChangeInLast3Frames >= 1000 && yChangeInLast3Frames < 2000)
						c3++;
					if(yChangeInLast3Frames >= 2000 && yChangeInLast3Frames < 3000)
						c4++;
					if(yChangeInLast3Frames >= 3000 && yChangeInLast3Frames < 4000)
						c5++;
					if(yChangeInLast3Frames >= 4000 && yChangeInLast3Frames < 5000)
						c6++;
					if(yChangeInLast3Frames >= 5000 && yChangeInLast3Frames < 6000)
						c7++;
					if(yChangeInLast3Frames >= 6000 && yChangeInLast3Frames < 7000)
						c8++;
					if(yChangeInLast3Frames >= 7000 && yChangeInLast3Frames < 8000)
						c9++;
					if(yChangeInLast3Frames >= 8000 && yChangeInLast3Frames < 9000)
						c10++;
					if(yChangeInLast3Frames >= 9000 && yChangeInLast3Frames < 10000)
						c11++;
					if(yChangeInLast3Frames >= 10000)
						c12++;
					if(yChangeInLast3Frames == 13)
						c13++;
					if(yChangeInLast3Frames == 14)
						c14++;
					if(yChangeInLast3Frames == 15)
						c15++;
					if(yChangeInLast3Frames == 16)
						c16++;
					if(yChangeInLast3Frames == 17)
						c17++;
					if(yChangeInLast3Frames == 18)
						c18++;
					if(yChangeInLast3Frames == 19)
						c19++;
					if(yChangeInLast3Frames >= 20 && yChangeInLast3Frames < 25)
						c20++;
					if(yChangeInLast3Frames >= 25 && yChangeInLast3Frames < 75)
						c21++;
					if(yChangeInLast3Frames >= 75 && yChangeInLast3Frames < 100)
						c22++;
					if(yChangeInLast3Frames >= 100)
						c23++;

				}
				}


				if(boundRect[zuordnungBackup[i]].br().x >= sequence.get(CV_CAP_PROP_FRAME_WIDTH)-1 || boundRect[zuordnungBackup[i]].tl().x <= 1)
					continue;

				//flaeche berechnen falls noch nicht geschehen
				if(data[i][5][0] == -1)
					data[i][5][0] = boundRect[zuordnungBackup[i]].width * boundRect[zuordnungBackup[i]].height;


				klassifikation(boundRect[zuordnungBackup[i]], i, data);

				/*//klassifizierung 
				float fahrzeug = 0.0;
				float mensch = 0.0;
				float fahrrad = 0.0;


				//x-aenderung mit einbeziehen
				if(neuerDurchschnittX2 < 10){
				fahrzeug += 0.71;
				mensch += 0.81;
				fahrrad += 0.57;
				}
				if(neuerDurchschnittX2 < 20){
				fahrzeug += 0.1;
				mensch += 0.09;
				fahrrad += 0.22;
				}
				if(neuerDurchschnittX2 < 30){
				fahrzeug += 0.04;
				mensch += 0.045;
				fahrrad += 0.125;
				}
				if(neuerDurchschnittX2 < 40){
				fahrzeug += 0.02;
				mensch += 0.01;
				fahrrad += 0.025;
				}
				if(neuerDurchschnittX2 < 50){
				fahrzeug += 0.02;
				mensch += 0.01;
				fahrrad += 0.025;
				}
				if(neuerDurchschnittX2 >= 50){
				fahrzeug += 0.07;
				mensch += 0.01;
				fahrrad += 0.025;
				}

				//breiten / hoehe verhaeltnis
				double relation = (double)((double)boundRect[zuordnungBackup[i]].width / (double)boundRect[zuordnungBackup[i]].height)*100;
				if(relation >= 170)
				fahrzeug += 0.73;
				if(relation >= 150 && relation < 170)
				fahrzeug += 0.05;
				if(relation >= 120 && relation < 150){
				fahrzeug += 0.07;
				fahrrad += 0.8;
				}
				if(relation < 120){
				fahrzeug += 0.13;
				fahrrad += 0.2;
				mensch += 0.4;
				}
				if(relation >= 120 && relation < 170)
				mensch += 0.6;
				if(relation < 50)
				mensch += 0.48;

				//y aenderung einbeziehn
				if(neuerDurchschnittY == 0){
				fahrzeug += 0.3;
				mensch += 0.22;
				fahrrad += 0.175;
				}
				if(neuerDurchschnittY == 1){
				fahrzeug += 0.18;
				mensch += 0.09;
				fahrrad += 0.15;
				}
				if(neuerDurchschnittY == 2){
				fahrzeug += 0.09;
				mensch += 0.07;
				fahrrad += 0.12;
				}
				if(neuerDurchschnittY > 2){
				fahrzeug += 0.03;
				mensch += 0.04;
				fahrrad += 0.05;
				}


				if(data[i][6] == 0){
				fahrzeug += 0.5;
				}
				if(data[i][6] == 1){
				mensch += 0.5;
				}
				if(data[i][6] == 2){
				fahrrad += 0.5;
				}

				if(fahrzeug >= mensch && fahrzeug >= fahrrad)
				data[i][6] = 0;
				else if (mensch >= fahrzeug && mensch >= fahrrad)
				data[i][6] = 1;
				else if (fahrrad >= fahrzeug && fahrrad >= mensch)
				data[i][6] = 2;
				*/}

			oldBoundRect = boundRect;
			oldContours = contours;

			trails = trails + 0.001 * phantom;

			erode(trails, check, Mat::ones(20, 20, CV_8UC1));
			dilate(check, check, Mat::ones(20, 25, CV_8UC1));




			int finish = cvGetTickCount();
			int duration = (int)(cvGetTickFrequency() * 1.0e6) / (finish - start);
			printStats(frame, "#" + std::to_string((int)sequence.get(CV_CAP_PROP_POS_FRAMES)) + " | " + std::to_string(duration) + " FPS");

			imshow("output | q or esc to quit | spacebar to pause | +/- to go to next/prev frame", frame);
			imshow("paths", check);

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
			sequence.set(CV_CAP_PROP_POS_FRAMES, sequence.get(CV_CAP_PROP_POS_FRAMES)-2); //geht 1 bild zur�ck
		}

		if (key  == '+'){
			repause = true;
			pause = false;
		}


		if (key == 'q' || key == 'Q' || key == 27)
			break;
	}

	std::cerr << "c0: " << c0 << std::endl;
	std::cerr << "c1: " << c1 << std::endl;
	std::cerr << "c2: " << c2 << std::endl;
	std::cerr << "c3: " << c3 << std::endl;
	std::cerr << "c4: " << c4 << std::endl;
	std::cerr << "c5: " << c5 << std::endl;
	std::cerr << "c6: " << c6 << std::endl;
	std::cerr << "c7: " << c7 << std::endl;
	std::cerr << "c8: " << c8 << std::endl;
	std::cerr << "c9: " << c9 << std::endl;
	std::cerr << "c10: " << c10 << std::endl;
	std::cerr << "c11: " << c11 << std::endl;
	std::cerr << "c12: " << c12 << std::endl;
	std::cerr << "c13: " << c13 << std::endl;
	std::cerr << "c14: " << c14 << std::endl;
	std::cerr << "c15: " << c15 << std::endl;
	std::cerr << "c16: " << c16 << std::endl;
	std::cerr << "c17: " << c17 << std::endl;
	std::cerr << "c18: " << c18 << std::endl;
	std::cerr << "c19: " << c19 << std::endl;
	std::cerr << "c20: " << c20 << std::endl;
	std::cerr << "c21: " << c21 << std::endl;
	std::cerr << "c22: " << c22 << std::endl;
	std::cerr << "c23: " << c23 << std::endl;
	std::cerr << "c24: " << c24 << std::endl;
	std::cerr << "c25: " << c25 << std::endl;
	std::cerr << "c26: " << c26 << std::endl;
	std::cerr << "c27: " << c27 << std::endl;
	std::cerr << "c28: " << c28 << std::endl;


	std::cerr << "cAll: " << cAll << std::endl;

	return 0;
}

