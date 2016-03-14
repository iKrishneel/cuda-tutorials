
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <omp.h>

/**
 * http://www.juergenwiki.de/work/wiki/doku.php?id=public%3ahog_descriptor_computation_and_visualization
 */

cv::Mat visualizeHOG(cv::Mat& src, /*std::vector<float>& */cv::Mat & descriptor_values,
                     cv::Size win_size, cv::Size cell_size, int scale_factor, double viz_factor) {   
    cv::Mat visual_image;
    cv::resize(src, visual_image, cv::Size(src.cols*scale_factor, src.rows*scale_factor));
    int gradientBinSize = 9;
    float radRangeForOneBin = 3.14/(float)gradientBinSize;
    int cells_in_x_dir = win_size.width / cell_size.width;
    int cells_in_y_dir = win_size.height / cell_size.height;
    int totalnrofcells = cells_in_x_dir * cells_in_y_dir;
    float*** gradient_strengths = new float**[cells_in_y_dir];
    int** cell_update_counter   = new int*[cells_in_y_dir];
    for (int y=0; y<cells_in_y_dir; y++) {
        gradient_strengths[y] = new float*[cells_in_x_dir];
        cell_update_counter[y] = new int[cells_in_x_dir];
        for (int x=0; x<cells_in_x_dir; x++) {
            gradient_strengths[y][x] = new float[gradientBinSize];
            cell_update_counter[y][x] = 0;
            for (int bin=0; bin<gradientBinSize; bin++)
                gradient_strengths[y][x][bin] = 0.0;
        }
    }
    int blocks_in_x_dir = cells_in_x_dir - 1;
    int blocks_in_y_dir = cells_in_y_dir - 1;
    
    int descriptor_data_idx = 0;
    int cellx = 0;
    int celly = 0;

#pragma omp parallel for num_threads(8) collapse(3)
    for (int blockx=0; blockx<blocks_in_x_dir; blockx++) {
        for (int blocky=0; blocky<blocks_in_y_dir; blocky++) {
            for (int cellNr=0; cellNr<4; cellNr++) {
                int cellx = blockx;
                int celly = blocky;
                if (cellNr==1) celly++;
                if (cellNr==2) cellx++;
                if (cellNr==3) {
                    cellx++;
                    celly++;
                }
                for (int bin=0; bin<gradientBinSize; bin++) {
                    // float gradient_strength = descriptor_values[descriptor_data_idx];
                    float gradient_strength = descriptor_values.at<float>(0, descriptor_data_idx);
                    descriptor_data_idx++;
                    gradient_strengths[celly][cellx][bin] += gradient_strength;
                }
                cell_update_counter[celly][cellx]++;
            }
        }
    }
#pragma omp parallel for num_threads(8) collapse(2)
    for (int celly=0; celly<cells_in_y_dir; celly++) {
        for (int cellx=0; cellx<cells_in_x_dir; cellx++) {
            float NrUpdatesForThisCell = (float)cell_update_counter[celly][cellx];
            for (int bin=0; bin<gradientBinSize; bin++) {
                gradient_strengths[celly][cellx][bin] /= NrUpdatesForThisCell;
            }
        }
    }
#pragma omp parallel for num_threads(8) collapse(2)
    for (int celly=0; celly<cells_in_y_dir; celly++) {
        for (int cellx=0; cellx<cells_in_x_dir; cellx++) {
            int drawX = cellx * cell_size.width;
            int drawY = celly * cell_size.height;
            int mx = drawX + cell_size.width/2;
            int my = drawY + cell_size.height/2;
            cv::rectangle(visual_image,
                          cv::Point(drawX*scale_factor,drawY*scale_factor),
                          cv::Point((drawX+cell_size.width)*scale_factor,
                                    (drawY+cell_size.height)*scale_factor),
                          CV_RGB(100,100,100),
                          1);
            for (int bin=0; bin<gradientBinSize; bin++) {
                float currentGradStrength = gradient_strengths[celly][cellx][bin];
                if (currentGradStrength==0)
                    continue;
                float currRad = bin * radRangeForOneBin + radRangeForOneBin/2;
 
                float dirVecX = cos( currRad );
                float dirVecY = sin( currRad );
                float maxVecLen = cell_size.width/2;
                float scale = viz_factor;
                float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;
                cv::line(visual_image,
                     cv::Point(x1*scale_factor,y1*scale_factor),
                     cv::Point(x2*scale_factor,y2*scale_factor),
                         CV_RGB(0,0,255),
                         1);
            }
        }
    }
    for (int y=0; y<cells_in_y_dir; y++) {
        for (int x=0; x<cells_in_x_dir; x++) {
            delete[] gradient_strengths[y][x];            
        }
        delete[] gradient_strengths[y];
        delete[] cell_update_counter[y];
    }
    delete[] gradient_strengths;
    delete[] cell_update_counter;
    return visual_image;
}

int main(int argc, char *argv[]) {

    // std::string filename = argv[1];
    // cv::Mat image = cv::imread(filename, 0);
    
    cv::VideoCapture cap(0);
    cv::Mat image;

    for (; ; ) {
        cap >> image;
        if (!image.empty()) {
            cv::resize(image, image, cv::Size(64, 128));
            cv::Mat src = image.clone();
            cv::cvtColor(image, image, CV_BGR2GRAY, 1);
            cv::cuda::GpuMat g_mat(image);

            // cv::cuda::resize(g_mat, g_mat, cv::Size(64, 128));
            // cv::cuda::cvtColor(g_mat, g_mat, CV_BGR2GRAY);
            
            
            cv::Ptr<cv::cuda::HOG> hog = cv::cuda::HOG::create();
            cv::cuda::GpuMat g_descriptors;
            hog->compute(g_mat, g_descriptors);

            cv::Mat descriptors;
            g_descriptors.download(descriptors);
            cv::Mat hog_img = visualizeHOG(src, descriptors,
                                           cv::Size(64, 128),
                                           cv::Size(8, 8), 10, 4);
            cv::imshow("image", image);
            cv::imshow("hog", hog_img);
            if(cv::waitKey(30) >= 0) {
                break;
            }
            
        }
    }
    return 0;
}

