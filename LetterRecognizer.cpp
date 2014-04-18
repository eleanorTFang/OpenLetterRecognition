#include <opencv2/opencv.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include "LetterImageCreator.h"
#include "LetterFeatureExtractor.h"
#include "LetterVoronoiSpecificer.h"
#include "LetterLabelCreator.h"
#include "LetterTrashCleaner.h"
#include "LetterIdentifier.h"
#include "LetterRecognizer.h"

using namespace cv;
using namespace letterrecog;

namespace {
    void drawInformationOfLabelToImage(const Rectangles& rects, const Values& similars, cv::Mat& view,
        const float& thresholdCorrel = 0.3f);
#ifdef _USING_WINDOW
    void on_mouse(int event, int x, int y, int flags, void* param);

    struct EventArgs {
        EventArgs(const LetterArgs* _args, LetterIdentificationer* _ident, Mat* _view) 
            : args(_args), ident(_ident), view(_view) {}
        const LetterArgs* args;
        LetterIdentificationer* ident;
        Mat* view;
    };
#endif
};

#ifdef _ENABLE_INVOCATION
int main(int argc, const char** argv)
{
    LetterArgs args;
    args.imageLetterText = argv[1];
    args.imageTargetName = argv[2];
    
    args.enableReverse = boost::lexical_cast<unsigned int>(argv[3]);
    args.enableCleaning = boost::lexical_cast<unsigned int>(argv[4]);
    args.enableErode = boost::lexical_cast<unsigned int>(argv[5]);
    args.enableDilate = boost::lexical_cast<unsigned int>(argv[6]);
    
    args.image.keypointMaxNum = boost::lexical_cast<unsigned int>(argv[7]);
    args.image.keypointRatio = boost::lexical_cast<float>(argv[8]);
    args.image.thresholdTrashDiameter = boost::lexical_cast<float>(argv[9]);
    args.image.thresholdKeypointRadius = boost::lexical_cast<float>(argv[10]);
    args.image.thresholdKeypointDistance = boost::lexical_cast<float>(argv[11]);
    args.image.thresholdImageCorrel = boost::lexical_cast<float>(argv[12]);

    recognizeLetter(args);
	return 0;
};
#endif

namespace {
    void recognizeLetter(const LetterArgs& args, const cv::Mat* letterImage)
    {
        // Creating letter image.
#ifdef _USING_QT4
        LetterImageCreator image(args.imageLetterText);
        const Mat& target = image.letterImage();
#else
        if (!letterImage) return;
        const Mat& target = *letterImage;
#endif

        Mat view = imread(args.imageTargetName, 1), source;
        Mat origin = imread(args.imageTargetName, 0);

        // Binarize the image
        adaptiveThreshold(origin, source, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 7, 8);
#ifdef _CREATE_MIDDLE_IMAGE
        imwrite("image/debug/middle-source.png", source);
#endif
        if (args.enableReverse == 1) source =~ source;
        if (args.enableCleaning == 1) {
            LetterTrashCleaner cleaner(source, args.image.thresholdTrashDiameter);
            source = cleaner.labelImage();
        }
        if (args.enableErode == 1) erode (source, source, Mat(), Point(-1, -1), 1);
        if (args.enableDilate == 1) dilate(source, source, Mat(), Point(-1, -1), 1);

        // Getting the featues of image
        LetterFeatureExtractor extractor(source, target, 1.2f, 5.0f, args.image.keypointMaxNum, &view);
        const Keypoints& keypoints = extractor.keypoints();

        LetterVoronoiSpecificer voronoi(source, keypoints, 20.0f);
        const Mat& labelImage = voronoi.labelImage();

        Mat resultImage = voronoi.drawEdges(view); 
        
        // Creating the rectangle of label
        LetterLabelCreator label(labelImage);
        const Rectangles& rects = label.rectangles();

        // Identification of letter
        LetterIdentificationer ident(target, source, args.imageLetterText, rects, args.image.keypointMaxNum);
        const Values& similars = ident.similars();

        // Drawing the rectangle of label
        drawInformationOfLabelToImage(rects, similars, resultImage, args.image.thresholdImageCorrel);

#ifdef _USING_WINDOW
        EventArgs event(&args, &ident, &resultImage);
        cvSetMouseCallback("LetterImage", &on_mouse, &event);
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


    void drawInformationOfLabelToImage(const Rectangles& rects, const Values& similars, cv::Mat& view, const float& thresholdCorrel)
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

#ifdef _USING_WINDOW
    void on_mouse(int event, int x, int y, int flags, void* param)
    {
        if (event == CV_EVENT_LBUTTONDOWN || event == CV_EVENT_RBUTTONDOWN) {
            EventArgs* letter = static_cast<EventArgs*>(param);
            const LetterArgs* args = static_cast<const LetterArgs*>(letter->args);
            LetterIdentificationer* ident = static_cast<LetterIdentificationer*>(letter->ident);
            Mat* view = static_cast<Mat*>(letter->view);

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
                    teach.at<float>(i, 0) = answer[i] = (args->image.thresholdImageCorrel < similars[i] ? 1.0f : 0.0f); 
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
#ifdef _VERBOSE
                    std::cout << "new " << (event == CV_EVENT_LBUTTONDOWN ? "correct" : "invalid") << " answer of " << (i + 1) << " is add" << std::endl;
#endif
                    // Traning
                    CvSVM svm;
                    CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
                    CvSVMParams param(CvSVM::C_SVC, CvSVM::RBF, 10.0, 8.0, 1.0, 10.0, 0.5, 0.1, NULL, criteria);
                    svm.train_auto(features, teach, Mat(), Mat(), param);
                    string saveFile = (boost::format("./model/%1%.xml") % ident->text()).str();
#ifdef _VERBOSE
                    std::cout << "svm model file is created on the " << saveFile << std::endl;
#endif
                    svm.save(saveFile.c_str());
                    drawInformationOfLabelToImage(rects, answer, *view);
                    imshow("LetterImage", *view);
                    break;
                }
            }            
        }
    }
#endif
    
}; // end of namespace



