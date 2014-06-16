#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <vector>
#include <iostream>
#include <stdlib.h>

using namespace cv;

std::string fileSrc = "C:/training/3/0000000000.png";

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
	vector<vector<int>> data; // geschwindigkeit x, geschwindigkeit y, durchschnittliche veränderung der geschwindigkeit in x richtung, durchschnittliche veränderung der geschwindigkeit in y richtung, frame in dem objekt gefunden, fläche zu beginn, klassifizierung 0 = auto, 1 = mensch, 2 = fahrrad

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

			cvtColor(frame, uniform, CV_BGR2HSV);
			split(uniform, channels);

			channels[2] = Mat::zeros(frame.size[0], frame.size[1], CV_8UC1);	// Helligkeit = 0

			merge(channels, uniform);

			mog.operator()(uniform, fore);
			mog.getBackgroundImage(back);

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

			std::vector<std::vector<Point> > contours;
			findContours(result, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

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

				// erase all objects that are thinner than a certain threshold
				// TODO: Exclude cases where Rectangle is thin because of cut off at the frame's edges
				if (boundRect[i].height / boundRect[i].width > 5)
				{
					contours.erase(contours.begin() + i);
					boundRect.erase(boundRect.begin() + i);
				}

				rectangle(frame, boundRect[i].tl(), boundRect[i].br(), 255, 2, 8, 0);

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
					
					//richtung prüfen
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
							if(currentRichtung > 0 && data[index][0] < 0 || currentRichtung < 0 && data[index][0] > 0)
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


				//konturen miteinander vergleichen und ähnlichste kontur finden
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

				if(data[index][5] == -1)
					continue;
				if(boundRect[zuordnung[i]].width * boundRect[zuordnung[i]].height >= (float)data[index][5])
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
				if(flaeche > data[index][5]*1.5)
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
				
				cv::Scalar color = cv::Scalar(128,128,128);
				

				if(zuordnungBackup.empty()){
					putText(frame, std::to_string(i), cvPoint(boundRect[zuordnung[i]].tl().x+50,boundRect[zuordnung[i]].tl().y+50),FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,0,250), 1, CV_AA);
					rectangle(frame, boundRect[zuordnung[i]].tl(), boundRect[zuordnung[i]].br(), color, 2, 8, 0);
					
					std::vector<std::vector<Point> > singleContour;
					singleContour.push_back(contours[zuordnung[i]]);
					drawContours(frame, singleContour, -1, Scalar(200, 255, 0), 1);
				}		
				else{
					for(unsigned int j = 0; j < zuordnungBackup.size(); j++){
						if(zuordnungBackup[j] == i){
							putText(frame, std::to_string(j), cvPoint(boundRect[zuordnung[i]].tl().x+50,boundRect[zuordnung[i]].tl().y+50),FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,0,250), 1, CV_AA);

							if(data[j][6] == 0)
								color = cv::Scalar(0,0,255);
							if(data[j][6] == 1)
								color = cv::Scalar(255,0,0);
							if(data[j][6] == 2)
								color = cv::Scalar(0,255,0);

							std::vector<std::vector<Point> > singleContour;
							singleContour.push_back(contours[zuordnung[i]]);
							drawContours(frame, singleContour, -1, Scalar(200, 255, 0), 1);
							rectangle(frame, boundRect[zuordnung[i]].tl(), boundRect[zuordnung[i]].br(), color, 2, 8, 0);
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
				if(((int)zuordnung.size())-1 < zuordnungBackup[i]){
					zuordnung2.push_back(-1);

					continue;
				}
					
				zuordnung2.push_back(zuordnung[zuordnungBackup[i]]);

				

			}
			


			//neue objekte zu zuordnung2 hinzufügen
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
			for(unsigned int i = 0; i < zuordnungBackup.size(); i++){
				
				//falls neues objekt
				if(i >= data.size()){

				//falls verlorenes object
				if(zuordnungBackup[i] == -1){
					vector<int> newData;
					newData.push_back(0); // x richtung
					newData.push_back(0); // y richtung
					newData.push_back(0); // durchschnitt veränderung x
					newData.push_back(0); // durchschnitt veränderung y
					newData.push_back(0); // frame 
					newData.push_back(0); // fläche
					newData.push_back(-1); // klassifizierung
					data.push_back(newData);
					continue;
				}

					vector<int> newData;
					int index = -1;
					for(unsigned int j = 0; j < zuordnung.size(); j++){
						if(zuordnung[j] == zuordnungBackup[i]){
							index = j;
							break;
						}
					}
					newData.push_back(boundRect[zuordnungBackup[i]].tl().x - oldBoundRect[index].tl().x); // x richtung
					newData.push_back(boundRect[zuordnungBackup[i]].tl().y - oldBoundRect[index].tl().y); // y richtung
					newData.push_back(0); // durchschnitt veränderung x
					newData.push_back(0); // durchschnitt veränderung y
					newData.push_back(sequence.get(CV_CAP_PROP_POS_FRAMES)); // frame

					
					if(boundRect[zuordnungBackup[i]].br().x >= sequence.get(CV_CAP_PROP_FRAME_WIDTH)-1 || boundRect[zuordnungBackup[i]].tl().x <= 1)
						newData.push_back(-1); // objekt außerhalb des bildes
					else
						newData.push_back(oldBoundRect[index].area()); // fläche
				
					newData.push_back(-1); // klassifizierung

					data.push_back(newData);
					continue;
				}
				
			//ansonsten updaten
				if(zuordnungBackup[i] == -1)
					continue;

					int index = -1;
					for(unsigned int j = 0; j < zuordnung.size(); j++){
						if(zuordnung[j] == zuordnungBackup[i]){
							index = j;
							break;
						}
					}
			
				int newRichtungX = boundRect[zuordnungBackup[i]].tl().x - oldBoundRect[index].tl().x;
				int newRichtungY = abs(boundRect[zuordnungBackup[i]].tl().y - oldBoundRect[index].tl().y);
				int alterDurchschnittX = data[i][2];
				int neuerDurchschnittX = (((alterDurchschnittX * ((sequence.get(CV_CAP_PROP_POS_FRAMES)-1) - data[i][4])) + (newRichtungX - data[i][0])) / (sequence.get(CV_CAP_PROP_POS_FRAMES) - data[i][4]));
				int neuerDurchschnittX2 = (((alterDurchschnittX * ((sequence.get(CV_CAP_PROP_POS_FRAMES)-1) - data[i][4])) + (abs(abs(newRichtungX) - data[i][0]))) / (sequence.get(CV_CAP_PROP_POS_FRAMES) - data[i][4]));
				int alterDurchschnittY = data[i][3];
				int neuerDurchschnittY = (((alterDurchschnittY * ((sequence.get(CV_CAP_PROP_POS_FRAMES)-1) - data[i][4])) + (abs(newRichtungY - data[i][1]))) / (sequence.get(CV_CAP_PROP_POS_FRAMES) - data[i][4]));
				data[i][0] = newRichtungX;
				data[i][1] = newRichtungY;
				data[i][2] = neuerDurchschnittX;
				data[i][2] = 0;
				data[i][3] = neuerDurchschnittY;

				//if(i == 211 || i == 252 || i == 294 || i == 236 ||  i == 265){
			

				
				if(boundRect[zuordnungBackup[i]].br().x >= sequence.get(CV_CAP_PROP_FRAME_WIDTH)-1 || boundRect[zuordnungBackup[i]].tl().x <= 1)
					continue;
				
				//fläche berechnen falls noch nicht geschehen
				if(data[i][5] == -1)
					data[i][5] = boundRect[zuordnungBackup[i]].width * boundRect[zuordnungBackup[i]].height;


				//klassifizierung 
				float fahrzeug = 0.0;
				float mensch = 0.0;
				float fahrrad = 0.0;


				//x-änderung mit einbeziehen
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

				//breiten / höhe verhältnis
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

				//y änderung einbeziehn
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
			}

			oldBoundRect = boundRect;
			oldContours = contours;

			

			int finish = cvGetTickCount();
			int duration = (int)(cvGetTickFrequency() * 1.0e6) / (finish - start);
			printStats(frame, "#" + std::to_string((int)sequence.get(CV_CAP_PROP_POS_FRAMES)) + " | " + std::to_string(duration) + " FPS");

			imshow("output | q or esc to quit | spacebar to pause | +/- to go to next/prev frame", frame);

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