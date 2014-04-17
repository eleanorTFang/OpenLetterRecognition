#include <vector>
#include <opencv2/opencv.hpp>

namespace letterrecog
{
    typedef std::vector<cv::KeyPoint> Keypoints;
    typedef std::vector<cv::Point> Vertices;

    class LetterVoronoiSpecificer
    {
    public:
        LetterVoronoiSpecificer(const cv::Mat& targetImage, const Keypoints& keypoints, const float& thresholdDistance = 30.0f);
        const cv::Mat& labelImage() { return labelImage_; }
        const cv::Mat drawEdges(const cv::Mat& image);

    private:
        const float thresholdDistance_;
        const cv::Mat targetImage_;
        const Keypoints keypoints_;

        cv::Mat labelImage_;
        Vertices sourceVertices_;
        Vertices targetVertices_;
    };

}; // end of namespace

