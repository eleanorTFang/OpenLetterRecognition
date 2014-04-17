#include <string>
#include <opencv2/opencv.hpp>

namespace letterrecog
{
    typedef std::vector<cv::KeyPoint> Keypoints;
    typedef std::vector<cv::Mat> Histgrams;
    typedef std::vector<cv::Rect> Rectangles;
    typedef std::vector<float> Values;
 
    class LetterIdentificationer
    {
    public:
        LetterIdentificationer(const cv::Mat& letterImage, const cv::Mat& sourceImage, const std::string& text, 
            const Rectangles& rects, const float& maxKeypoints = 5000.0f);
        const Values& similars() { return similars_; }

    private:
        const cv::Mat letterImage_;
        const cv::Mat sourceImage_;
        const Rectangles rects_;
        const float maxKeypoints_;

        Histgrams histgrams_;
        Values similars_;
    };

}; // end of namespace

 
