#include <opencv2/opencv.hpp>
#include "LetterFineExclusioner.h"

using namespace cv;

namespace letterrecog 
{
    LetterFineExclusioner::LetterFineExclusioner(const cv::Mat& labelImage, const float& areaMax)
        : labelImage_(labelImage), areaMax_(areaMax), resultImage_(cv::Mat(Size(labelImage.cols, labelImage.rows), CV_8UC1, Scalar::all(0)))
    {
#ifdef _CREATE_MIDDLE_IMAGE
        imwrite("image/debug/middle-label-source.png", labelImage);
#endif
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours(labelImage, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0));

        for (int i = 0; i < contours.size(); i++) {
            const Mat figure(contours[i]);
            const double area = contourArea(figure);
            if (areaMax_ < area) {
                drawContours(resultImage_, contours, i, CV_RGB(255, 255, 255), CV_FILLED, CV_AA, hierarchy, 0, Point());
            }
        }
#ifdef _CREATE_MIDDLE_IMAGE
        imwrite("image/debug/middle-label-result.png", resultImage_);
#endif
    }

}; // end of namespace



