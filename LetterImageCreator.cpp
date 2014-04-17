#include <QTextCodec>
#include <opencv2/opencv.hpp>
#include "LetterImageCreator.h"

using namespace cv;

namespace letterrecog 
{
    LetterImageCreator::LetterImageCreator(const std::string text, const unsigned int fontSize)
        : letterImage_(Mat(Size(320, 40), CV_8UC3, Scalar::all(0))), fontSize_(fontSize),
            blackLetter_(fontQt(FONT_NAME, fontSize_, Scalar(0, 0, 0), CV_FONT_BOLD, CV_STYLE_NORMAL, 0)),
            whiteLetter_(fontQt(FONT_NAME, fontSize_, Scalar(255, 255, 255), CV_FONT_BOLD, CV_STYLE_NORMAL, 0))
    {
        // Creating the fonts.
        QTextCodec::setCodecForCStrings(QTextCodec::codecForLocale()); 
        cvNamedWindow("LetterImage");

        // Drawing the letter on image.
#ifdef _DOUBLE_LETTER
        cv::rectangle(letterImage_, Point(0, 40), Point(320, 80), Scalar(255, 255, 255), CV_FILLED);
        addText(letterImage_, text.c_str(), Point(10, 70), blackLetter_);
#endif
        addText(letterImage_, text.c_str(), Point(10, 30), whiteLetter_);

#ifdef _CREATE_MIDDLE_IMAGE
        imwrite("image/debug/middle-letter.png", letterImage_);
#endif
    }
   
}; // end of namespace


