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

    // Roberts kernel, kx is one diagonal and ky is the other
    Mat kx = (Mat_<float>(2, 2) << 1, 0, 0, -1);
    Mat ky = (Mat_<float>(2, 2) << 0, 1, -1, 0);

    filter2D(grayF, fx, CV_32F, kx, Point(0, 0), 0, BORDER_REPLICATE);
    filter2D(grayF, fy, CV_32F, ky, Point(0, 0), 0, BORDER_REPLICATE);
}

// --------------------------------------------------------
// Gradient Vector Flow (GVF) computation
// --------------------------------------------------------
void computeGVF(const Mat& fx, const Mat& fy,
    Mat& u, Mat& v,
    float mu = 0.2f,
    float dt = 0.1f,
    int iterations = 100)
{
    CV_Assert(fx.type() == CV_32F && fy.type() == CV_32F);
    CV_Assert(fx.size() == fy.size());

    Mat mag, g2;
    magnitude(fx, fy, mag);

    //|∇f|^2
    g2 = mag.mul(mag);

    u = fx.clone();
    v = fy.clone();

    // Discrete 2D laplacian kernel
    Mat laplacianKernel = (Mat_<float>(3, 3) <<
        0, 1, 0,
        1, -4, 1,
        0, 1, 0
        );

    // laplacian of u and v
    Mat lapU, lapV;
    Mat tmpU, tmpV;

    for (int it = 0; it < iterations; ++it)
    {
        filter2D(u, lapU, CV_32F, laplacianKernel, Point(-1, -1), 0, BORDER_REPLICATE);
        filter2D(v, lapV, CV_32F, laplacianKernel, Point(-1, -1), 0, BORDER_REPLICATE);

        tmpU = u - fx;
        tmpU = mu * lapU - tmpU.mul(g2);

        tmpV = v - fy;
        tmpV = mu * lapV - tmpV.mul(g2);

        u += dt * tmpU;
        v += dt * tmpV;
    }
}

// --------------------------------------------------------
// Draw Gradient Flow Vector field
// --------------------------------------------------------
Mat drawGVFField(const Mat& image, const Mat& u, const Mat& v,
    int step = 8, float scale = 5.0f, float magnitudeThreshold = 1e-3f)
{
    CV_Assert(u.type() == CV_32F && v.type() == CV_32F);
    CV_Assert(u.size() == v.size());

    Mat vis;
    if (image.channels() == 1)
        cvtColor(image, vis, COLOR_GRAY2BGR);
    else
        vis = image.clone();

    for (int y = 0; y < vis.rows; y += step)
    {
        for (int x = 0; x < vis.cols; x += step)
        {
            float vx = u.at<float>(y, x);
            float vy = v.at<float>(y, x);
            float magnitude = std::sqrt(vx * vx + vy * vy);
            if (magnitude < magnitudeThreshold) continue;

            float nx = vx / magnitude;
            float ny = vy / magnitude;

            Point p0(x, y);
            Point p1(x + (int)(scale * nx), y + (int)(scale * ny));

            arrowedLine(vis, p0, p1, Scalar(255, 0, 0), 1, LINE_AA, 0, 0.3);
        }
    }
    return vis;
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
    Mat equalizedLuminance;
    equalizeHist(gray, equalizedLuminance);

    // Robert gradients(Equalize(L(I)))
    Mat fx, fy;
    computeRobertsGradient(equalizedLuminance, fx, fy);

    // Gradient Vector Flow (GVF)
    Mat u, v;
    computeGVF(fx, fy, u, v, 0.2f, 0.1f, 100);

    imshow("Original", img);
    imshow("GVF Field (arrows) raster image", drawGVFField(img, u, v));
    imshow("GVF Field (arrows) equalized luminance", drawGVFField(equalizedLuminance, u, v));

    waitKey(0);
    return 0;
}