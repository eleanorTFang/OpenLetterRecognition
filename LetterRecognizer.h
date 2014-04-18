#include <opencv2/opencv.hpp>

namespace {

    struct ImageArgs {
        unsigned int keypointMaxNum;
        float keypointRatio;
        
        float thresholdTrashDiameter;
        float thresholdKeypointRadius;
        float thresholdKeypointDistance;
        float thresholdImageCorrel;
    };

    struct LetterArgs {
        std::string imageLetterText;
        std::string imageTargetName;
        
        unsigned int enableReverse;
        unsigned int enableCleaning;
        unsigned int enableErode;
        unsigned int enableDilate;

        ImageArgs image;
    };

    void recognizeLetter(const LetterArgs& args, const cv::Mat* letterImage = NULL);

}; // end of namespace

