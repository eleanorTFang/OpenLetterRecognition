#include <opencv2/opencv.hpp>

namespace letterrecog
{
    typedef std::vector<cv::Rect> Rectangles;
    typedef std::vector<float> Values;
        
    class LetterLabelCreator
    {
    public:
        LetterLabelCreator(const cv::Mat& labelImage, const float& areaMin = 3000.0f, const float& areaMax = 10000.0f,
            const float& circleMin = 0.2f, const float& circleMax = 0.6f);

        const Rectangles& rectangles() const { return rects_; }
        const Values& circles() const { return circles_; }
        const Values& areas() const { return areas_; }

    private:
        Rectangles rects_;
        Values areas_;
        Values circles_;

        const float areaMin_;
        const float areaMax_;
        const float circleMin_;
        const float circleMax_;
    };

}; // end of namespace

