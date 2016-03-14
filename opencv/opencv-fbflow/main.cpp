#include <iostream>
#include <vector>
#include <sstream>

#include <opencv2/dnn.hpp>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"

#include <omp.h>

using namespace std;
using namespace cv;
using namespace cv::cuda;

template <typename T>
inline T mapVal(T x, T a, T b, T c, T d)
{
    x = ::max(::min(x, b), a);
    return c + (d-c) * (x-a) / (b-a);
}

static void colorizeFlow(const Mat &u, const Mat &v, Mat &dst)
{
    double uMin, uMax;
    cv::minMaxLoc(u, &uMin, &uMax, 0, 0);
    double vMin, vMax;
    cv::minMaxLoc(v, &vMin, &vMax, 0, 0);
    uMin = ::abs(uMin); uMax = ::abs(uMax);
    vMin = ::abs(vMin); vMax = ::abs(vMax);
    float dMax = static_cast<float>(::max(::max(uMin, uMax), ::max(vMin, vMax)));

    dst.create(u.size(), CV_8UC3);

#pragma omp parallel for collapse(2) num_threads(8) shared(dst)
    for (int y = 0; y < u.rows; ++y)
    {
        for (int x = 0; x < u.cols; ++x)
        {
            dst.at<uchar>(y,3*x) = 0;
            dst.at<uchar>(y,3*x+1) = (uchar)mapVal(-v.at<float>(y,x), -dMax, dMax, 0.f, 255.f);
            dst.at<uchar>(y,3*x+2) = (uchar)mapVal(u.at<float>(y,x), -dMax, dMax, 0.f, 255.f);
        }
    }
}

cv::Mat prev_image;
int proc_counter = 0;

int main(int argc, char **argv)
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        return -1;
    }

    bool running = true, gpuMode = true;
    int64 t, t0=0, t1=1, tc0, tc1;

    cv::Mat frameR;
    cv::Mat frameL;
    Ptr<cuda::FarnebackOpticalFlow> d_calc = cuda::FarnebackOpticalFlow::create();
    for(;;) {
        cap >> frameR;
        if (frameR.empty()) {
            return -1;
        }
        cv::resize(frameR, frameR, cv::Size(640, 480));
        cv::cvtColor(frameR, frameR, CV_BGR2GRAY, 1);
        
        if (!prev_image.empty()) {

            // cout << prev_image.size() << ", " << frameR.size()  << "\n";
            
            cv::cuda::GpuMat d_frameL(prev_image);
            cv::cuda::GpuMat d_frameR(frameR);
            cv::cuda::GpuMat d_flow;
            cv::Mat flowxy, flowx, flowy, image;
            t = getTickCount();

            if (gpuMode)
            {
                tc0 = getTickCount();
                d_calc->calc(d_frameL, d_frameR, d_flow);
                tc1 = getTickCount();

                GpuMat planes[2];
                cuda::split(d_flow, planes);

                planes[0].download(flowx);
                planes[1].download(flowy);
            }
            else
            {
                tc0 = getTickCount();
                calcOpticalFlowFarneback(prev_image, frameR, flowxy, d_calc->getPyrScale(),
                                         d_calc->getNumLevels(), d_calc->getWinSize(),
                                         d_calc->getNumIters(), d_calc->getPolyN(),
                                         d_calc->getPolySigma(), d_calc->getFlags());
                tc1 = getTickCount();

                Mat planes[] = {flowx, flowy};
                split(flowxy, planes);
                flowx = planes[0]; flowy = planes[1];
            }

            colorizeFlow(flowx, flowy, image);

            stringstream s;
            s << "mode: " << (gpuMode?"GPU":"CPU");
            putText(image, s.str(), Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255,0,255), 2);

            s.str("");
            s << "opt. flow FPS: " << cvRound((getTickFrequency()/(tc1-tc0)));
            putText(image, s.str(), Point(5, 65), FONT_HERSHEY_SIMPLEX, 1., Scalar(255,0,255), 2);

            s.str("");
            s << "total FPS: " << cvRound((getTickFrequency()/(t1-t0)));
            putText(image, s.str(), Point(5, 105), FONT_HERSHEY_SIMPLEX, 1., Scalar(255,0,255), 2);

            imshow("flow", image);

            char ch = (char)waitKey(3);
            if (ch == 27)
                running = false;
            else if (ch == 'm' || ch == 'M')
                gpuMode = !gpuMode;

            t0 = t;
            t1 = getTickCount();
            

            if(cv::waitKey(30) >= 0) {
                break;
            }
        }
        prev_image = frameR.clone();
    }
    
    
    return 0;
}
