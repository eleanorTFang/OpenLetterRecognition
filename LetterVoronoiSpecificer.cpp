#include <opencv2/opencv.hpp>
#include "LetterVoronoiSpecificer.h"

using namespace cv;

namespace letterrecog 
{
    LetterVoronoiSpecificer::LetterVoronoiSpecificer(const cv::Mat& targetImage, const Keypoints& keypoints,
        const float& thresholdDistance)
    
        : targetImage_(targetImage), keypoints_(keypoints), thresholdDistance_(thresholdDistance)
    {
        std::vector<Point2f> vertices(keypoints_.size());
        Subdiv2D subdiv;
        subdiv.initDelaunay(cv::Rect(0, 0, targetImage_.cols, targetImage_.rows));

        unsigned int i = 0;
        for (Keypoints::const_iterator iter = keypoints_.begin(); iter != keypoints_.end(); ++iter) {
            vertices[i++] = Point2f(iter->pt.x, iter->pt.y);
        }
        subdiv.insert(vertices);

        std::vector<Vec4f> edgeList;
        subdiv.getEdgeList(edgeList);
        labelImage_ = Mat(Size(targetImage_.cols, targetImage_.rows), CV_8U, Scalar::all(0));

        for(std::vector<cv::Vec4f>::iterator edge = edgeList.begin(); edge != edgeList.end(); edge++) {
            const cv::Point p1(edge->val[0], edge->val[1]);
            const cv::Point p2(edge->val[2], edge->val[3]);

            if (0 <= edge->val[0] && edge->val[0] < targetImage_.cols &&
                0 <= edge->val[1] && edge->val[1] < targetImage_.rows &&
                0 <= edge->val[2] && edge->val[2] < targetImage_.cols &&
                0 <= edge->val[3] && edge->val[3] < targetImage_.rows) {

                const double distance = sqrt(pow(edge->val[0] - edge->val[2], 2) + pow(edge->val[1] - edge->val[3], 2));        
                int edges, vertexs, edget, vertext;
                subdiv.locate(Point2f(edge->val[0], edge->val[1]), edges, vertexs);
                subdiv.locate(Point2f(edge->val[2], edge->val[3]), edget, vertext);

                if (distance < thresholdDistance_) {
                    line(labelImage_, p1, p2, Scalar(255, 255, 255));
                    sourceVertices_.push_back(p1);
                    targetVertices_.push_back(p2);
                }
            }
        }
    }


    const cv::Mat LetterVoronoiSpecificer::drawEdges(const cv::Mat& image)
    {
        cv::Mat result(image);
        for (unsigned int i = 0; i < sourceVertices_.size(); ++i) {
            line(result, sourceVertices_[i], targetVertices_[i], Scalar(255, 255, 255));
        }
        return result;
    }


}; // end of namespace


