#include <opencv2/opencv.hpp>

using namespace cv;

static void LocalThBinarization(Mat qr, Mat &out) {
    Mat qrf;
    qr.convertTo(qrf, CV_32F);

    Mat mean;
    blur(qrf, mean, Size(qr.cols / 8 + 1,qr.rows / 8 + 1));

    Mat mean2;
    blur(qrf.mul(qrf), mean2, Size(qr.cols / 8 + 1,qr.rows / 8 + 1));

    Mat sigma;
    cv::sqrt(mean2 - mean.mul(mean), sigma);

    Mat localTh = mean.mul(0.2 * (sigma / (qr.cols / 4) - 1) + 1);

    out = Mat(qr.size(), CV_8U);
    for (int i = 0; i < qr.cols; i++) {
        for (int j = 0; j < qr.rows; j++) {
            out.at<uchar>(i,j) = (qr.at<uchar>(i,j) < localTh.at<float>(i,j) ? 0 : 255);
        }
    }
}

static void LocalPreWorkGray (Mat &qrGray) {
    int minp = 256;
    for (int i = 0; i < qrGray.cols; i++) {
        for (int j = 0; j < qrGray.rows; j++) {
            if (qrGray.at<uchar>(i,j) < minp) {
                minp = qrGray.at<uchar>(i,j);
            }
        }
    }
    for (int i = 0; i < qrGray.cols; i++) {
        for (int j = 0; j < qrGray.rows; j++) {
            qrGray.at<uchar>(i,j) -= minp;
        }
    }
}
