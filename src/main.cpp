#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>

#define sqr(x) ((x) * (x))

using namespace std;
using namespace cv;

enum {UP, RIGHT, DOWN, LEFT};

vector<Vec4i> hierarchy;
vector<vector<Point>> contours;
Mat frame;
int f [1000000];

double crossProduct (Point2f A, Point2f B, Point2f C) {
    return ((B.x - A.x) * (C.y - B.y) - (C.x - B.x) * (B.y - A.y));
}

int countHierarchy (int x) {
    // return the maxium deepth
    if (~f[x]) return f[x];
    int res = 0;
    for (int sx = hierarchy[x][2]; ~sx; sx = hierarchy[sx][2])
        res = max (res, countHierarchy (sx) + 1);
    f[x] = res;
    return res;
}

vector<int> findCandidates (Mat frame) {
    vector<int> res;
    res.clear ();

    for (int i = 0; i < contours.size (); ++i) f[i] = -1;
    
    for (int i = 0; i < contours.size (); ++i)
        if (countHierarchy (i) >= 5) res.push_back (i);

    return res;
}

double dist (Point2f x, Point2f y) {
    // Return the distance between two points
    return sqrt (sqr (x.x - y.x) + sqr (x.y - y.y));
}

double distLine (Point2f X, Point2f A, Point2f B) {
    // Returen the distance between Point X and Line AB
    return abs (crossProduct (X, A, B) / dist (A, B));
}

vector<int> getPoint (double AB, double BC, double CA, int A, int B, int C) {
    vector<int> res;
    res.clear ();
    if (AB > BC && AB > CA) {
        res.push_back (C);
        res.push_back (A);
        res.push_back (B);
    }
    if (BC > AB && BC > CA) {
        res.push_back (A);
        res.push_back (B);
        res.push_back (C);
    }
    if (CA > AB && CA > BC) {
        res.push_back (B);
        res.push_back (A);
        res.push_back (C);
    }
    return res;
}

bool dist_constraint (double AB, double BC, double CA) {
    // TODO : add more constraint
    if (AB > BC && AB > CA) {
        if (abs (BC - CA) > 0.2 * max (BC, CA)) return 1;
        return 0;
    }
    if (BC > AB && BC > CA) {
        if (abs (AB - CA) > 0.2 * max (CA, AB)) return 1;
        return 0;
    }
    if (CA > AB && CA > BC) {
        if (abs (BC - AB) > 0.2 * max (BC, AB)) return 1;
        return 0;
    }
    return 1;
}

bool area_constraint (double areaA, double areaB, double areaC) {
    double areaMean = (areaA + areaB + areaC) / 3;
    double sigma = sqr (areaA - areaMean) + sqr (areaB - areaMean) + sqr (areaC - areaMean);
    sigma /= 3 * sqr (areaMean);
    return sigma > 0.005;
}

Point2f findAwayFromLine (int x, Point2f A, Point2f B) {
    // Find a Point in contours[x] that most away from line AB 
    Point2f res;
    double maxP = 0;
    for (int i = 0; i < contours[x].size(); ++i) {
        double tmp = distLine (contours[x][i], A, B);
        if (tmp > maxP) {
            maxP = tmp;
            res = contours[x][i];
        } 
    }
    return res;
}

Point2f findAwayFromPoint (int x, Point2f P) {
    // Find a Point in contours[x] that most away from point P
    Point2f res;
    double maxP = 0;
    for (int i = 0; i < contours[x].size(); ++i) {
        double tmp = dist (contours[x][i], P);
        if (tmp > maxP) {
            maxP = tmp;
            res = contours[x][i];
        } 
    }
    return res;
}

Point2f findN (Point2f top, Point2f right, Point2f left) {
        return Point2f (right.x + left.x - top.x, right.y + left.y - top.y);
}

vector<Point2f> findCorners (int top, int left, int right, Point2f meanTop, Point2f meanLeft, Point2f meanRight) {
    vector<Point2f> res;
    res.clear ();
    Point2f A = findAwayFromLine (top, meanLeft, meanRight);
    Point2f verA = findAwayFromPoint (top, A);
    Point2f B = findAwayFromLine (right, A, verA);
    Point2f C = findAwayFromLine (left, A, verA);
    res.push_back (A);
    res.push_back (B);
    res.push_back (C);
    res.push_back (findN (A, B, C));
    return res;
}

vector<Mat> findQR (vector<int> candidates) {
    vector<Mat> res; res.clear ();
    Mat qr, raw, gray, thres, rawFrame;
    qr = Mat::zeros(100, 100, CV_8UC1);
    raw = Mat::zeros(100, 100, CV_8UC1);
    rawFrame = frame;
    int size = candidates.size ();
    if (size < 3) return res; 
    vector<Moments> mome(size);
    vector<Point2f> mean(size);
    vector<Point2f> pts1, pts2;
    // Caculate the mean point
    for (int i = 0; i < size; ++i) {
        mome[i] = moments (contours[candidates[i]], false); 
        mean[i] = Point2f (mome[i].m10/mome[i].m00 , mome[i].m01/mome[i].m00);
    }
    for (int A = 0; A < size; ++A) 
        for (int B = A + 1; B < size; ++B)
            for (int C = B + 1; C < size; ++C) {
                double AB = dist (mean[A],mean[B]);
                double BC = dist (mean[B],mean[C]);
                double CA = dist (mean[C],mean[A]);
                if (dist_constraint (AB, BC, CA)) continue;
                if (area_constraint (contourArea (contours[candidates[A]]), 
                            contourArea (contours[candidates[B]]), 
                            contourArea (contours[candidates[C]]))
                        ) 
                    continue;
                vector<int> tmp = getPoint (AB, BC, CA, A, B, C);
                int top = tmp[0]; int left = tmp[1]; int right = tmp[2];
                // Use cross product to determine left and right
                if (crossProduct (mean[left], mean[top], mean[right]) < 0) 
                    swap (left, right);
                // Draw the FIG
                drawContours (frame, contours, candidates[top] , Scalar(255,0,0), 2, 8, hierarchy, 0 );
                drawContours (frame, contours, candidates[left] , Scalar(0,255,0), 2, 8, hierarchy, 0 );
                drawContours (frame, contours, candidates[right] , Scalar(0,0,255), 2, 8, hierarchy, 0 );
                // Find all corners
                pts1 = findCorners (candidates[top], candidates[left], candidates[right], mean[top], mean[left], mean[right]);
                /*
                circle (frame, cvPoint (pts1[0].x, pts1[0].y), 10 , Scalar(255,0,0), 2, 8, 0);
                circle (frame, cvPoint (pts1[2].x, pts1[2].y), 10, Scalar(0,255,0), 2, 8, 0);
                circle (frame, cvPoint (pts1[1].x, pts1[1].y), 10, Scalar(0,0,255), 2, 8, 0);
                circle (frame, cvPoint (pts1[3].x, pts1[3].y), 10, Scalar(255,255,255), 2, 8, 0);
                */
                pts2.clear ();
                pts2.push_back(Point2f(0,0));
				pts2.push_back(Point2f(qr.cols,0));
				pts2.push_back(Point2f(0, qr.rows));
				pts2.push_back(Point2f(qr.cols, qr.rows));
                // Do perspective transform
                Mat M = getPerspectiveTransform (pts1,pts2);
                warpPerspective (rawFrame, raw, M, Size (qr.cols,qr.rows));
                copyMakeBorder (raw, qr, 10, 10, 10, 10, BORDER_CONSTANT, Scalar(255,255,255));
                res.push_back (qr);
                return res;
            }
    return res;
}

int main(int argc, const char *argv[]) {
    
	VideoCapture capture(0);

    capture >> frame;

	Mat gray(frame.size(), CV_MAKETYPE(frame.depth(), 1));
	Mat traces(frame.size(), CV_8UC3);
	Mat edges (frame.size (), CV_MAKETYPE (frame.depth (), 1));

    cout << "Press any key to return." << endl;

    for (int key = -1; !~key; capture >> frame) {
        // Change to grayscale
        cvtColor (frame, gray, CV_RGB2GRAY);

        // Find FIG candidates
        Canny(frame, edges, 100 , 200, 3);		// Apply Canny edge detection on the gray image
        findContours (edges, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        vector<int> candidates = findCandidates (gray);
        
        vector<Mat> qrs = findQR (candidates);

		imshow ("Image", frame);
        for (int i = 0; i < qrs.size (); ++i) {
            String Name = "QR";
            String id;
            stringstream sout;
            sout << i;
            id = sout.str ();
            Name += id;
            imshow (Name, qrs[i]);
        }

        key = waitKey (100);
    }

    return 0;
}
