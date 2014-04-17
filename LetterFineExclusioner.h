#include <opencv2/opencv.hpp>

namespace letterrecog
{
    typedef std::vector<cv::Rect> Rectangles;
    typedef std::vector<float> Values;
        
    class LetterFineExclusioner
    {
    public:
        LetterFineExclusioner(const cv::Mat& labelImage, const float& areaMax = 10000.0f);
        cv::Mat labelImage() { return resultImage_; }

    private:
        const cv::Mat labelImage_;
        const float areaMax_;
        cv::Mat resultImage_;
    };

}; // end of namespace

