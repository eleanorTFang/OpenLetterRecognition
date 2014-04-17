#include <opencv2/opencv.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include "LetterImageCreator.h"
#include "LetterFeatureExtractor.h"
#include "LetterVoronoiSpecificer.h"
#include "LetterLabelCreator.h"
#include "LetterFineExclusioner.h"
#include "LetterIdentificationer.h"

using namespace cv;
using namespace letterrecog;

void recogniteLetter(const int argc, const char** argv);
void drawInformationOfLabelToImage(const Rectangles& rects, const Values& similars, cv::Mat& view);
void on_mouse(int event, int x, int y, int flags, void* param);


int main(int argc, const char** argv)
{
    recogniteLetter(argc, argv);
	return 0;
}


void recogniteLetter(const int argc, const char** argv)
{
    const unsigned int r = boost::lexical_cast<unsigned int>(argv[3]);
    const unsigned int e = boost::lexical_cast<unsigned int>(argv[4]);
    const unsigned int d = boost::lexical_cast<unsigned int>(argv[5]);
    const unsigned int m = boost::lexical_cast<unsigned int>(argv[6]);
    const unsigned int f = boost::lexical_cast<unsigned int>(argv[7]);

    const string text(argv[2]);
    LetterImageCreator image(text);
    const Mat& target = image.letterImage();

    Mat view = imread(argv[1], 1), source;
    Mat origin = imread(argv[1], 0);

    // Binarize the image
    adaptiveThreshold(origin, source, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 7, 8);
#ifdef _CREATE_MIDDLE_IMAGE
    imwrite("../../image/debug/middle-source.png", source);
#endif
    if (r == 1) source =~ source;
    if (f == 1) {
        LetterFineExclusioner exclusioner(source, 5);
        source = exclusioner.labelImage();
    }
    if (e == 1) erode (source, source, Mat(), Point(-1, -1), 1);
    if (d == 1) dilate(source, source, Mat(), Point(-1, -1), 1);

    // Getting the featues of image
    LetterFeatureExtractor extractor(source, target, 1.2f, 5.0f, m, &view);
    const Keypoints& keypoints = extractor.keypoints();

    LetterVoronoiSpecificer voronoi(source, keypoints, 20.0f);
    const Mat& labelImage = voronoi.labelImage();

    Mat resultImage = voronoi.drawEdges(view); 
    
    // Creating the rectangle of label
    LetterLabelCreator label(labelImage);
    const Rectangles& rects = label.rectangles();

    // Identification of letter
    LetterIdentificationer ident(target, source, text, rects, m);
    const Values& similars = ident.similars();

    // Drawing the rectangle of label
    drawInformationOfLabelToImage(rects, similars, resultImage);

    cvSetMouseCallback("LetterImage", &on_mouse, &ident);
    cv::imshow("LetterImage", resultImage);
#ifdef _CREATE_MIDDLE_IMAGE
    imwrite("../../image/debug/middle-result.png", source);
#endif

    while (true) {
        int key = cv::waitKey(0);
        if (key == 27) break;
    }

}


void drawInformationOfLabelToImage(const Rectangles& rects, const Values& similars, cv::Mat& view)
{
    static const float thresholdCorrel = 0.3f;    

    // Drawing the information of label
    const CvFont blue   = fontQt(FONT_NAME, 12, Scalar(  0,   0, 128), CV_FONT_BOLD, CV_STYLE_NORMAL, 0);
    const CvFont green  = fontQt(FONT_NAME, 12, Scalar(  0, 255, 255), CV_FONT_BOLD, CV_STYLE_NORMAL, 0);
    for (int i = 0; i < rects.size(); ++i) {
        cv::rectangle(view, rects[i].tl(), rects[i].br(), (similars[i] > thresholdCorrel ? CV_RGB(0, 255, 255) : CV_RGB(0, 0, 128)), 2, CV_AA);
    }
    for (int i = 0; i < rects.size(); ++i) {
        addText(view, (boost::format("%1%") % similars[i]).str(), 
            Point(rects[i].tl().x, rects[i].tl().y - 5), (similars[i] > thresholdCorrel ? green : blue));
    }

}


void on_mouse(int event, int x, int y, int flags, void* param)
{
    if (event == CV_EVENT_LBUTTONDOWN) {
        std::cout << "@ Left mouse button pressed at: " << x << "," << y << std::endl;
        LetterIdentificationer* ident = static_cast<LetterIdentificationer*>(param);
    }
}

