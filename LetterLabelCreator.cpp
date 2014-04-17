#include <opencv2/opencv.hpp>
#include "LetterLabelCreator.h"

using namespace cv;

namespace letterrecog 
{
    LetterLabelCreator::LetterLabelCreator(const cv::Mat& labelImage, const float& areaMin, const float& areaMax, const float& circleMin, const float& circleMax)
        : areaMin_(areaMin), areaMax_(areaMax), circleMin_(circleMin), circleMax_(circleMax)
    {
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours(labelImage, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0));

        const Mat filledImage(Size(labelImage.cols, labelImage.rows), CV_8UC1, Scalar::all(0));
        for (int i = 0; i < contours.size(); i++) {
            drawContours(filledImage, contours, i, CV_RGB(255, 255, 255), CV_FILLED);
        }
        erode (filledImage, filledImage, Mat(), Point(-1, -1), 1);
        dilate(filledImage, filledImage, Mat(), Point(-1, -1), 1);
#ifdef _CREATE_MIDDLE_IMAGE
        imwrite("../../image/debug/middle-labeling.png", filledImage);
#endif
        findContours(filledImage, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
        for (int i = 0; i < contours.size(); i++) {
            const Mat figure(contours[i]);
            const double area = contourArea(figure);
            const double perimeter = arcLength(figure, true);
            const double circle = 4.0 * CV_PI * area / (perimeter * perimeter);

            const Rect rect = boundingRect(figure);
            const double rectArea = rect.area();

            rects_.push_back(rect);
            areas_.push_back(area);
            circles_.push_back(circle);
        }
    }

}; // end of namespace



