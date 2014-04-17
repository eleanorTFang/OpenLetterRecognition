#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "LetterIdentifier.h"

using namespace cv;
using namespace boost::filesystem;

namespace letterrecog
{
    LetterIdentificationer::LetterIdentificationer(const cv::Mat& letterImage, const cv::Mat& sourceImage, const std::string& text, 
        const Rectangles& rects, const float& maxKeypoints)
        : letterImage_(letterImage), sourceImage_(sourceImage), rects_(rects), maxKeypoints_(maxKeypoints), text_(text)
    {
        initModule_nonfree();

        const unsigned int size = rects.size();
        vector<Mat> images(size + 1);
        images[0] = letterImage;
        for (unsigned int i = 0; i < size; ++i) {
            images[i + 1] = sourceImage(rects[i]);
        }

        // Select feature detector and extractor.
        const SiftFeatureDetector detector(maxKeypoints_);
        const SiftDescriptorExtractor extractor;

        Keypoints keypoints; Mat descriptors;
        detector.detect(images[0], keypoints);
        const unsigned int cluster = keypoints.size(); 

        // sourceImage is letter-image, target image is natual image
        BOWKMeansTrainer bowTrainer(cluster, TermCriteria(CV_TERMCRIT_ITER, 100, 0.001), 1, KMEANS_PP_CENTERS);

        // Creating the featues on image.
        vector< vector<KeyPoint> > keys(size + 1);
        for (unsigned int i = 0; i < (size + 1); ++i) {
            detector.detect(images[i], keypoints);
            keys[i] = keypoints;
            if (0 < keypoints.size()) {
                extractor.compute(images[i], keypoints, descriptors);
                bowTrainer.add(descriptors);
            }
        }
        const Mat vocabulary = bowTrainer.cluster();

        // Clustering      
        Ptr<DescriptorExtractor> sift = new SiftDescriptorExtractor();
        Ptr<DescriptorMatcher> flann = DescriptorMatcher::create("FlannBased");

        BOWImgDescriptorExtractor bowExtractor(sift, flann);
        bowExtractor.setVocabulary(vocabulary);

        // Creating histgram
        histgrams_.resize(size + 1);
        similars_.resize(size + 1);
        for (unsigned int i = 0; i < (size + 1); ++i) {
            Mat histgram;
            bowExtractor.compute(images[i], keys[i], histgram);
            histgrams_[i] = histgram;
        }

        const string xmlPath = (boost::format("./model/%1%.xml") % text_).str();    
        const path path(xmlPath.c_str());
        boost::system::error_code error;
        const bool result = exists(path, error);
        if (result && !error) {
            // Identification using svm
            CvSVM svm;
            svm.load(xmlPath.c_str());
            for (unsigned int i = 0; i < (size + 1); ++i) {
                double correl = -1.0f;
                if (0 < histgrams_[i].rows) {
                    correl = svm.predict(histgrams_[i]);
                }
                similars_[i] = correl;
            } 
        } else {
            // Original identification without svm
            for (unsigned int i = 0; i < (size + 1); ++i) {
                double correl = 0.0f;
                if (histgrams_[0].type() == histgrams_[i].type() && histgrams_[0].type() == CV_32F) {
                    correl = compareHist(histgrams_[0], histgrams_[i], CV_COMP_CORREL);
                }
                similars_[i] = correl;
            }
        }
    }

}; // end of namespace


