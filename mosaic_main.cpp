#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main() {
    cout << "OpenCV version: " << CV_VERSION << endl;

    Mat img = imread("luffy.jpg");
    if (img.empty())
    {
        cerr << "Could not load image luffy.jpg" << endl;
        return -1;
    }

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);    // L(I) <- Luminance(I)

    // In the paper: equalize luminance
    Mat grayEq;
    equalizeHist(gray, grayEq);             // Equalize(L(I))

    imshow("Original", img);
    imshow("Gray", gray);
    imshow("grayEq", grayEq);

    waitKey(0);
    return 0;
}
