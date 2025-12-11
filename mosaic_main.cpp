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
// Draw Gradient Vector Flow field
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

// --------------------------------------------------------
// Simple 3x3 non-maximum suppression on the image
// --------------------------------------------------------
void nonMaxSuppression3x3(const Mat& inputImage, Mat& outputImage)
{
    CV_Assert(inputImage.type() == CV_32F);
    outputImage = Mat::zeros(inputImage.size(), CV_32F);

    for (int y = 1; y < inputImage.rows - 1; ++y)
    {
        const float* prev = inputImage.ptr<float>(y - 1);
        const float* curr = inputImage.ptr<float>(y);
        const float* next = inputImage.ptr<float>(y + 1);
        float* out = outputImage.ptr<float>(y);

        for (int x = 1; x < inputImage.cols - 1; ++x)
        {
            float val = curr[x];
            bool isMax = true;

            for (int dy = -1; dy <= 1 && isMax; ++dy)
            {
                const float* rowPtr = (dy == -1 ? prev : (dy == 0 ? curr : next));
                for (int dx = -1; dx <= 1; ++dx)
                {
                    if (dy == 0 && dx == 0) continue;
                    if (rowPtr[x + dx] >= val)
                    {
                        isMax = false;
                        break;
                    }
                }
            }

            if (isMax)
                out[x] = val;
        }
    }
}

// --------------------------------------------------------
// Place tiles on the image
// --------------------------------------------------------
bool placeTile(Mat& mosaic, Mat& occupied,
    const Mat& sourceColor,
    Point2f center,
    float angleRadian,
    int tileSize)
{
    float angleDegree = angleRadian * 180.0f / (float)CV_PI;

    // Define rotated square
    RotatedRect rotatedRectangle(center, Size2f((float)tileSize, (float)tileSize), angleDegree);
    Point2f tileCornersFloat[4];
    rotatedRectangle.points(tileCornersFloat);

    // Convert to integer polygon
    vector<Point> tilePolygon(4);
    for (int i = 0; i < 4; ++i)
        tilePolygon[i] = Point(cvRound(tileCornersFloat[i].x), cvRound(tileCornersFloat[i].y));

    Rect tileBoundingBox = boundingRect(tilePolygon);

    // Check bounds
    if (tileBoundingBox.x < 0 || tileBoundingBox.y < 0 ||
        tileBoundingBox.x + tileBoundingBox.width  > mosaic.cols ||
        tileBoundingBox.y + tileBoundingBox.height > mosaic.rows)
        return false;

    // Check overlap
    for (int y = tileBoundingBox.y; y < tileBoundingBox.y + tileBoundingBox.height; ++y)
    {
        for (int x = tileBoundingBox.x; x < tileBoundingBox.x + tileBoundingBox.width; ++x)
        {
            if (occupied.at<uchar>(y, x))
            {
                if (pointPolygonTest(tilePolygon, Point2f((float)x, (float)y), false) >= 0)
                    return false; // overlaps existing tile
            }
        }
    }

    // Compute tile color as mean in a small square around the center
    int halfTileSize = tileSize / 2;
    Rect tileRegion(
        (int)center.x - halfTileSize, 
        (int)center.y - halfTileSize, 
        tileSize, tileSize);
    tileRegion &= Rect(0, 0, sourceColor.cols, sourceColor.rows);
    Scalar meanColor = mean(sourceColor(tileRegion));

    // Draw tile
    fillConvexPoly(mosaic, tilePolygon, meanColor);

    // Mark occupied
    for (int y = tileBoundingBox.y; y < tileBoundingBox.y + tileBoundingBox.height; ++y)
    {
        for (int x = tileBoundingBox.x; x < tileBoundingBox.x + tileBoundingBox.width; ++x)
        {
            if (pointPolygonTest(tilePolygon, Point2f((float)x, (float)y), false) >= 0)
                occupied.at<uchar>(y, x) = 255;
        }
    }

    return true;
}

int main() {

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

    // GVF magnitude
    Mat gvfMagnitude;
    magnitude(u, v, gvfMagnitude);

    Mat nonMaxSuppressedMagnitude;
    nonMaxSuppression3x3(gvfMagnitude, nonMaxSuppressedMagnitude);

    // Thresholds high and low
    double minVal, maxVal;
    minMaxLoc(nonMaxSuppressedMagnitude, &minVal, &maxVal);
    float thresholdHigh = (float)(0.3 * maxVal);  
    float thresholdLow = (float)(0.15 * maxVal);

    // Collect seed points (nonMaxSuppressedMagnitude > thresholdHigh)
    vector<Point> seeds;
    for (int y = 0; y < nonMaxSuppressedMagnitude.rows; ++y)
    {
        const float* row = nonMaxSuppressedMagnitude.ptr<float>(y);
        for (int x = 0; x < nonMaxSuppressedMagnitude.cols; ++x)
        {
            if (row[x] > thresholdHigh)
                seeds.emplace_back(x, y);
        }
    }

    // Sort seeds by decreasing strength (nonMaxSuppressedMagnitude value)
    sort(seeds.begin(), seeds.end(),
        [&nonMaxSuppressedMagnitude](const Point& a, const Point& b)
        {
            return nonMaxSuppressedMagnitude.at<float>(a.y, a.x) > nonMaxSuppressedMagnitude.at<float>(b.y, b.x);
        });

    // Prepare mosaic canvas and occupancy mask
    const int tileSize = 3;
    Mat mosaic(img.size(), img.type(), Scalar(255, 255, 255));
    Mat occupied(img.size(), CV_8U, Scalar(0));

    // Phase 1: place tiles at seed points
    for (const Point& p : seeds)
    {
        float ux = u.at<float>(p.y, p.x);
        float vy = v.at<float>(p.y, p.x);
        float vectorMagnitude = std::sqrt(ux * ux + vy * vy);
        if (vectorMagnitude < 1e-4f) continue;

        float tileAngle = atan2(vy, ux);
        placeTile(mosaic, occupied, img, Point2f((float)p.x, (float)p.y),
            tileAngle, tileSize);
    }

    // Phase 2: fill remaining regions by scanning image
    for (int y = 0; y < img.rows; y += tileSize / 2)
    {
        for (int x = 0; x < img.cols; x += tileSize / 2)
        {
            float val = nonMaxSuppressedMagnitude.at<float>(y, x);

            float tileAngle;
            if (val < thresholdLow)
            {
                // really flat area -> place tiles horizontally
                tileAngle = 0.0f;
            }
            else
            {
                // near an edge -> follow GVF
                float ux = u.at<float>(y, x);
                float vy = v.at<float>(y, x);
                float vectorMagnitude = std::sqrt(ux * ux + vy * vy);

                if (vectorMagnitude < 1e-6f)
                    tileAngle = 0.0f;
                else
                    tileAngle = atan2(vy, ux);
            }

            placeTile(mosaic, occupied, img,
                Point2f((float)x, (float)y),
                tileAngle, tileSize);
        }
    }

    // --- Coverage stats ---
    int totalPixels = img.rows * img.cols;
    int coveredPixels = countNonZero(occupied);
    double coveragePct = 100.0 * (double)coveredPixels / (double)totalPixels;
    double uncoveredPct = 100.0 - coveragePct;

    cout << "Total pixels     : " << totalPixels << endl;
    cout << "Covered pixels   : " << coveredPixels << endl;
    cout << "Coverage         : " << coveragePct << " %" << endl;
    cout << "Uncovered (gaps) : " << uncoveredPct << " %" << endl;

    imshow("Original", img);
    imshow("GVF Field (arrows) raster image", drawGVFField(img, u, v));
    imshow("GVF Field (arrows) equalized luminance", drawGVFField(equalizedLuminance, u, v));
    imshow("Mosaic", mosaic);

    waitKey(0);
    return 0;
}
