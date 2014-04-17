#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "LetterFeatureExtractor.h"

using namespace cv;

namespace letterrecog 
{
    LetterFeatureExtractor::LetterFeatureExtractor(const cv::Mat& targetImage, const cv::Mat& letterImage, 
        const float& thresholdRatio, const float& thresholdDistance, const float& maxKeypoints, cv::Mat* sourceImage) 
        : targetImage_(targetImage), letterImage_(letterImage), 
            thresholdRatio_(thresholdRatio), thresholdDistance_(thresholdDistance), maxKeypoints_(maxKeypoints), sourceImage_(sourceImage)
    {
        initModule_nonfree();

        // Select feature detector and extractor.
        const SiftFeatureDetector detector(maxKeypoints_);
        const SiftDescriptorExtractor extractor;

        Keypoints keypointsA, keypointsB;
        Mat descriptorsA, descriptorsB;

        // Creating the featues on image.
        detector.detect(targetImage, keypointsA);
        extractor.compute(targetImage, keypointsA, descriptorsA);
        detector.detect(letterImage, keypointsB);
        extractor.compute(letterImage, keypointsB, descriptorsB);

        // Matching the featues.
        vector< vector<DMatch> > matches;
        BFMatcher matcher;
        matcher.knnMatch(descriptorsA, descriptorsB, matches, 2);

        // Filtering the good featues only.
        // We get many features as possible. Then we narrow it down.
        vector<DMatch> matches_good;
        for (int i = 0; i < matches.size(); ++i) {
            if (matches[i][0].distance < thresholdRatio_ * matches[i][1].distance) {
                matches_good.push_back(matches[i][0]);
            }
        }

        // Filtering the good keypoints only.
        keypoints_.resize(matches_good.size());
        for (int i = 0; i < matches_good.size(); ++i) {
            keypoints_[i] = keypointsA[matches_good[i].queryIdx];
        }
        // Filterin the letter featues only.
        // Local features of the characters are dense. Therefore, features away disabled.
        for (Keypoints::iterator iter = keypoints_.begin(); iter != keypoints_.end();) {
            if (iter->size > thresholdDistance_) {
                iter = keypoints_.erase(iter);
                if (iter == keypoints_.end()) break;
           } else {
                ++iter;
           }
        }

        // Keeping the match ratio.
        targetRatio_ = keypointsA.size() / keypoints_.size();
        letterRatio_ = keypointsB.size() / keypoints_.size();

#ifdef _CREATE_MIDDLE_IMAGE
        if (sourceImage_) {
            Mat result;
            drawKeypoints(*sourceImage_, keypoints_, result, CV_RGB(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            imwrite("./image/debug/middle-keypoint.png", result);
        }
#endif
    }

}; // end of namespace

    
