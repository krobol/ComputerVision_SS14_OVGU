#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{

	VideoCapture sequence("C:/Users/Tom/Pictures/training/6/0000000000.png");
	if (!sequence.isOpened())
	{
		cerr << "Failed to open Image Sequence!\n" << endl;
		return 1;
	}

	Mat image;
	namedWindow("Image | q or esc to quit", CV_WINDOW_NORMAL);

	for (;;)
	{
		sequence >> image;
		if (image.empty())
		{
			cout << "End of Sequence" << endl;
			break;
		}

		imshow("image | q or esc to quit", image);

		char key = (char)waitKey(50);
		if (key == 'q' || key == 'Q' || key == 27)
			break;
	}

	return 0;
}