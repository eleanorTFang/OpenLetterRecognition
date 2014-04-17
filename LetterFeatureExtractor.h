#include <vector>
#include <opencv2/opencv.hpp>

namespace letterrecog
{
    typedef std::vector<cv::KeyPoint> Keypoints;

    class LetterFeatureExtractor
    {
    public: 
        LetterFeatureExtractor(const cv::Mat& targetImage, const cv::Mat& letterImage, 
            const float& thresholdRatio = 1.2f, const float& thresholdDistance = 5.0f, 
            const float& maxKeypoints = 5000, cv::Mat* sourceImage = NULL);

        const Keypoints& keypoints() const { return keypoints_; }
        const float& targetRatio() const { return targetRatio_; }
        const float& letterRatio() const { return letterRatio_; }

    private:
        const float thresholdRatio_;
        const float thresholdDistance_;
        const float maxKeypoints_;
        const cv::Mat targetImage_;
        const cv::Mat letterImage_;
        const cv::Mat* sourceImage_;

        Keypoints keypoints_;        
        float targetRatio_;        
        float letterRatio_;        
    };        

}; // end of namespace

