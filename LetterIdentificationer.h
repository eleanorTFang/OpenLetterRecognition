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
        LetterIdentificationer(const cv::Mat& letterImage, const cv::Mat& sourceImage, 
            const std::string& text, const Rectangles& rects, const float& maxKeypoints = 5000.0f);

        const Rectangles& rectangles() const { return rects_; }
        const Histgrams& histgrams() const { return histgrams_; }
        const Values& similars() const { return similars_; }
        const std::string& text() const { return text_; }

        void similar(const unsigned int index, const float& value) {
            similars_[index] = value;
        }            

    private:
        const cv::Mat letterImage_;
        const cv::Mat sourceImage_;
        const Rectangles rects_;
        const float maxKeypoints_;
        const std::string text_;
        
        Histgrams histgrams_;
        Values similars_;
    };

}; // end of namespace

 
