
#include <iostream>
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

void drawLines(cv::Mat& dst, const std::vector<cv::Vec2f>& lines) {
    dst.setTo(cv::Scalar::all(0));
    for (size_t i = 0; i < lines.size(); ++i) {
        float rho = lines[i][0], theta = lines[i][1];
        cv::Point pt1, pt2;
        double a = std::cos(theta), b = std::sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        cv::line(dst, pt1, pt2, cv::Scalar::all(255));
    }
}

int main(int argc, char *argv[]) {

    // std::string file = std::string(argv[1]);
    // cv::Mat img = cv::imread(file);
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        return -1;
    }
    
    for (; ; ) {
        cv::Mat img;
        cap >> img;

        if (img.empty()) {
            return -1;
        }
        cv::resize(img, img, cv::Size(640, 480));
        cv::cvtColor(img, img, CV_BGR2GRAY);
        cv::Mat outimg, outimg2;

        cv::cuda::GpuMat d_src, d_dst;
        d_src.upload(img);
        cv::Ptr<cv::cuda::CannyEdgeDetector> canny =
            cv::cuda::createCannyEdgeDetector(50.0, 100.0);
        canny->detect(d_src, d_dst);
        d_dst.download(outimg);
        
        cv::cuda::GpuMat hline_src, hline_dst;
        hline_src.upload(outimg);
        cv::Ptr<cv::cuda::HoughLinesDetector> hough =
            cv::cuda::createHoughLinesDetector(1.0f, 1.5 * M_PI/180.0f, 100);
        hough->detect(hline_src, hline_dst);
        // hline_dst.download(hough_lines);
        std::vector<cv::Vec2f> lines;
        hough->downloadResults(hline_dst, lines);

        std::cout << "NUMBER OF LINES: " << lines.size()  << "\n";
        
        cv::Mat hough_lines = cv::Mat::zeros(img.size(), CV_8UC1);
        drawLines(hough_lines, lines);
        
        cv::imshow("image", img);
        cv::imshow("canny", outimg);
        cv::imshow("lines", hough_lines);
        
        if (cv::waitKey(30) >= 0) {
            cap.release();
            break;
        }
    }

    // print device info
    cv::cuda::DeviceInfo info;
    std::cout << "THREADS:" << info.maxThreadsPerBlock()  << "\n";
    
    return 0;
}










