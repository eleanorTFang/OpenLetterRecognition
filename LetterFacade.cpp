#include <opencv2/opencv.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include "LetterImageCreator.h"
#include "LetterFeatureExtractor.h"
#include "LetterVoronoiSpecificer.h"
#include "LetterLabelCreator.h"
#include "LetterFineExclusioner.h"
#include "LetterIdentifier.h"

using namespace cv;
using namespace letterrecog;

namespace {
    static const float thresholdCorrel = 0.3f;    
};

namespace {
    void recogniteLetter(const int argc, const char** argv, const Mat* letterImage = NULL);
    void drawInformationOfLabelToImage(const Rectangles& rects, const Values& similars, cv::Mat& view);
    void on_mouse(int event, int x, int y, int flags, void* param);
};

namespace {
    struct EventArgs {
        EventArgs(LetterIdentificationer* _ident, Mat* _view) {
            ident = _ident;
            view = _view;
        }
        LetterIdentificationer* ident;
        Mat* view;
    }; 
};

int main(int argc, const char** argv)
{
    recogniteLetter(argc, argv);
	return 0;
}


namespace {
    void recogniteLetter(const int argc, const char** argv, const Mat* letterImage)
    {
        const unsigned int r = boost::lexical_cast<unsigned int>(argv[3]);
        const unsigned int e = boost::lexical_cast<unsigned int>(argv[4]);
        const unsigned int d = boost::lexical_cast<unsigned int>(argv[5]);
        const unsigned int m = boost::lexical_cast<unsigned int>(argv[6]);
        const unsigned int f = boost::lexical_cast<unsigned int>(argv[7]);

        // Creating letter image.
        const string text(argv[2]);
#ifdef _USING_QT4
        LetterImageCreator image(text);
        const Mat& target = image.letterImage();
#else
        const Mat target = *letterImage; 
#endif

        Mat view = imread(argv[1], 1), source;
        Mat origin = imread(argv[1], 0);

        // Binarize the image
        adaptiveThreshold(origin, source, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 7, 8);
#ifdef _CREATE_MIDDLE_IMAGE
        imwrite("image/debug/middle-source.png", source);
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

#ifdef _USING_WINDOW
        EventArgs args(&ident, &resultImage);
        cvSetMouseCallback("LetterImage", &on_mouse, &args);
        cv::imshow("LetterImage", resultImage);
#ifdef _CREATE_MIDDLE_IMAGE
        imwrite("image/debug/middle-result.png", source);
#endif
        while (true) {
            int key = cv::waitKey(0);
            if (key == 27) break;
        }
        destroyAllWindows();
#endif

    }


    void drawInformationOfLabelToImage(const Rectangles& rects, const Values& similars, cv::Mat& view)
    {
        // Drawing the information of label
        for (int i = 0; i < rects.size(); ++i) {
            cv::rectangle(view, rects[i].tl(), rects[i].br(), (similars[i + 1] > thresholdCorrel ? CV_RGB(0, 255, 255) : CV_RGB(0, 0, 128)), 2, CV_AA);
        }
#ifdef _USING_QT4
        const CvFont blue  = fontQt(FONT_NAME, 12, Scalar(  0,   0, 128), CV_FONT_BOLD, CV_STYLE_NORMAL, 0);
        const CvFont green = fontQt(FONT_NAME, 12, Scalar(  0, 255, 255), CV_FONT_BOLD, CV_STYLE_NORMAL, 0);
        for (int i = 0; i < rects.size(); ++i) {
            addText(view, (boost::format("%1%") % similars[i + 1]).str(), 
                Point(rects[i].tl().x, rects[i].tl().y - 5), (similars[i + 1] > thresholdCorrel ? green : blue));
        }
#endif
    }


    void on_mouse(int event, int x, int y, int flags, void* param)
    {
        if (event == CV_EVENT_LBUTTONDOWN || event == CV_EVENT_RBUTTONDOWN) {
            EventArgs* args = static_cast<EventArgs*>(param);
            LetterIdentificationer* ident = static_cast<LetterIdentificationer*>(args->ident);
            Mat* view = static_cast<Mat*>(args->view);

            // Getting features of rectangles.
            const Histgrams& histgrams = ident->histgrams();
            const Rectangles& rects = ident->rectangles();
            const Values& similars = ident->similars();
            if (histgrams.size() <= 0) return;           
 
            const unsigned int cols = histgrams[0].cols;
            const unsigned int rows = histgrams.size();

            Mat features = Mat::zeros(Size(cols, rows), CV_32FC1);
            Mat teach(Size(1, similars.size()), CV_32FC1, Scalar(-1));    

            // fratures and similar are copied.
            Values answer(similars.size());
            for (unsigned int i = 0; i < rows; ++i) {
                if (0 < histgrams[i].rows) {
                    teach.at<float>(i, 0) = answer[i] = (thresholdCorrel < similars[i] ? 1.0f : 0.0f); 
                    for (unsigned int j = 0; j < cols; ++j) {
                        features.at<float>(i, j) = histgrams[i].at<float>(0, j);
                    }
                }
            }

            // selected rectangle is correct.
            for (unsigned int i = 0; i < rects.size(); ++i) {
                if ((rects[i].x <= x && x <= (rects[i].x + rects[i].width)) &&
                    (rects[i].y <= y && y <= (rects[i].y + rects[i].height))) {
                    teach.at<float>(i + 1, 0) = answer[i + 1] = (event == CV_EVENT_LBUTTONDOWN ? 1.0f : -1.0f); // selected rectangle is correct.
                    ident->similar(i + 1, answer[i + 1]);
                    std::cout << "new " << (event == CV_EVENT_LBUTTONDOWN ? "correct" : "invalid") << " answer is add:" << (i + 1) << std::endl;
                    // Traning
                    CvSVM svm;
                    CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
                    CvSVMParams param(CvSVM::C_SVC, CvSVM::RBF, 10.0, 8.0, 1.0, 10.0, 0.5, 0.1, NULL, criteria);
                    svm.train_auto(features, teach, Mat(), Mat(), param);
                    svm.save((boost::format("./model/%1%.xml") % ident->text()).str().c_str());
                    drawInformationOfLabelToImage(rects, answer, *view);
                    imshow("LetterImage", *view);
                    break;
                }
            }            
        }
    }

}; // end of namespace



