#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace letterrecog
{
    typedef std::vector<cv::KeyPoint> Keypoints;

    const static char* FONT_NAME = "ヒラギノ角ゴ ProN";

    class LetterImageCreator
    {
    public:
        LetterImageCreator(const std::string text, const unsigned int fontSize = 32);
        const cv::Mat& letterImage() const { return letterImage_; }
        const CvFont& whiteLetter() const { return whiteLetter_; }
        const CvFont& blackLetter() const { return whiteLetter_; }

    private:
        const unsigned int fontSize_;
        const CvFont whiteLetter_;
        const CvFont blackLetter_;
        cv::Mat letterImage_;
    };

}; // end of namespace

