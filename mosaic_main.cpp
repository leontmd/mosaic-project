#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// --------------------------------------------------------
// Roberts gradient on equalized luminance
// G(I): Roberts Gradient(Equalize(L(I)))
// --------------------------------------------------------
void computeRobertsGradient(const Mat& grayEq, Mat& fx, Mat& fy)
{
    Mat grayF;
    grayEq.convertTo(grayF, CV_32F, 1.0f / 255.0f);

    Mat kx = (Mat_<float>(2, 2) << 1, 0,
        0, -1);
    Mat ky = (Mat_<float>(2, 2) << 0, 1,
        -1, 0);

    filter2D(grayF, fx, CV_32F, kx, Point(0, 0), 0, BORDER_REPLICATE);
    filter2D(grayF, fy, CV_32F, ky, Point(0, 0), 0, BORDER_REPLICATE);
}

int main() {
    cout << "OpenCV version: " << CV_VERSION << endl;

    Mat img = imread("luffy.jpg");
    if (img.empty())
    {
        cerr << "Could not load image luffy.jpg" << endl;
        return -1;
    }

    // L(I) = Luminance(I)
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);    

    // Equalize(L(I))
    Mat grayEq;
    equalizeHist(gray, grayEq);

    // Robert gradients(Equalize(L(I)))
    Mat fx, fy;
    computeRobertsGradient(grayEq, fx, fy);

    imshow("Original", img);
    imshow("Gray", gray);
    imshow("grayEq", grayEq);

    waitKey(0);
    return 0;
}