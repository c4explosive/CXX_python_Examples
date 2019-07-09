#include "Detector.hpp"

Detector::Detector(const cv::Mat& img)
{
    this->img = img.clone();
}

void Detector::forward(cv::Mat &cno)
{
     cv::dnn::Net cnn = cv::dnn::readNetFromTensorflow(
      "frozen_inference_graph.pb", "graph.pbtxt");
    //The model is in: https://github.com/tensorflow/models/tree/master/research/object_detection

    /*if( img.cols > 1000 || img.rows > 1000)
        cv::resize(img, img, cv::Size(1000, cvRound(
            1000*((double) img.rows / (double) img.cols))));*/

    cnn.setInput(cv::dnn::blobFromImage(img, 1.0, cv::Size(300, 300),
            cv::Scalar(), true, false));
    cno = cnn.forward();
}

void Detector::tensorIterator(const cv::Mat &cno,
                               std::vector<cv::Mat>& detections)
{
    #define LN 7
    int cols = img.cols, rows = img.rows;
    float * detect = (float *) cno.data;
    for(size_t l=0; l < cno.total(); l+=LN)
    {
        float* ins = detect+l;
        float score = (float) *(ins+2);
        int type = (int) *(ins+1);

        if(score > 0.5)
        {
            int left = cvRound( *(ins+3) * (float) cols);
            int top = cvRound( *(ins+4) * (float) rows);
            int right = cvRound( *(ins+5) * (float) cols);
            int bottom = cvRound( *(ins+6) * (float) rows);

            cv::Vec6d stmpVect({(double) left, 
                            (double) top, (double) right, (double) bottom, 
                            (double) type, (double) score});
           
            cv::Mat vtmpRecRes(stmpVect);

            detections.push_back(vtmpRecRes.t());

        }
    }

}

void Detector::detect(cv::Mat& mdetections)
{
    cv::Mat cno;
    this->forward(cno);
    std::vector<cv::Mat> detections;
    this->tensorIterator(cno, detections);

    auto dsize = detections.size();

    if(dsize > 0)
    {
        cv::Mat tmpTensor(dsize, 6, CV_32F);
        int i = 0;
        for(auto &dect: detections)
        {
            dect.copyTo(tmpTensor(cv::Range(i,i+1), cv::Range::all()));
            i++;
        }
        mdetections = tmpTensor.clone();
    }
    else
    {
        mdetections = cv::Mat();
    }

}

void Detector::operator()(cv::Mat& mdetections)
{
    this->detect(mdetections);
}

Detector::~Detector()
{
}
