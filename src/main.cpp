#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace cv;

enum {UP, RIGHT, DOWN, LEFT};

vector<Vec4i> hierarchy;
vector<vector<Point>> contours;
Mat frame;
int f [100000];

bool crossProduct (Point2f A, Point2f B, Point2f C) {
    return ((B.x - A.x) * (C.y - B.y) - (C.x - B.x) * (B.y - A.y) < 0);
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

float dist (Point2f x, Point2f y) {
    // Return the distance between two points
    return sqrt ((x.x - y.x) * (x.x - y.x) + (x.y - y.y) * (x.y - y.y));
}

vector<int> getPoint (float AB, float BC, float CA, int A, int B, int C) {
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

bool dist_constraint (float AB, float BC, float CA) {
    // TODO : add more constraint
    if (AB > BC && AB > CA) {
        if (abs (BC - CA) > 0.2 * max (BC, CA)) return 1;
    }
    if (BC > AB && BC > CA) {
        if (abs (AB - CA) > 0.2 * max (CA, AB)) return 1;
    }
    if (CA > AB && CA > BC) {
        if (abs (BC - AB) > 0.2 * max (BC, AB)) return 1;
    }
}

Point2f findUP (int id) {
    Point2f res (frame.cols, frame.rows);
    for (int i = 0; i < contours[id].size (); ++i) 
        if (contours[id][i].y < res.y || (contours[id][i].y == res.y && contours[id][i].x < res.x)) 
            res = contours[id][i];
    return res;
}

Point2f findRIGHT (int id) {
    Point2f res (0, frame.rows);
    for (int i = 0; i < contours[id].size (); ++i) 
        if (contours[id][i].x > res.x || (contours[id][i].x == res.x && contours[id][i].y < res.y)) 
            res = contours[id][i];
    return res;
}

Point2f findDOWN (int id) {
    Point2f res (0, 0);
    for (int i = 0; i < contours[id].size (); ++i) 
        if (contours[id][i].y > res.y || (contours[id][i].y == res.y && contours[id][i].x > res.x)) 
            res = contours[id][i];
    return res;
}

Point2f findLEFT (int id) {
    Point2f res (frame.cols, 0);
    for (int i = 0; i < contours[id].size (); ++i) 
        if (contours[id][i].x < res.x || (contours[id][i].y == res.y && contours[id][i].y > res.y)) 
            res = contours[id][i];
    return res;
}

Point2f findN (Point2f top, Point2f right, Point2f left) {
    return Point2f (right.x + left.x - top.x, right.y + left.y - top.y);
}

vector<Point2f> findCorners (int top, int left, int right, int orientation) {
    vector<Point2f> res;
    res.clear ();
    switch (orientation) {
        case UP:
            res.push_back (findUP (top));
            res.push_back (findRIGHT (right));
            res.push_back (findLEFT (left));
            break;
        case RIGHT:
            res.push_back (findRIGHT (top));
            res.push_back (findDOWN (right));
            res.push_back (findUP (left));
            break;
        case DOWN:
            res.push_back (findDOWN (top));
            res.push_back (findLEFT (right));
            res.push_back (findRIGHT (left));
            break;
        case LEFT:
            res.push_back (findLEFT (top));
            res.push_back (findUP (right));
            res.push_back (findDOWN (left));
            break;
    }
    res.push_back (findN (res[0], res[1], res[2]));
    swap (res[2], res[3]);
    return res;
}

int findOrientation (Point2f meanA, Point2f meanB, Point2f meanC) {
    if (meanA.y < meanB.y && meanA.y < meanC.y) return UP;
    if (meanA.x > meanB.x && meanA.x > meanC.x) return RIGHT;
    if (meanA.y > meanB.y && meanA.y > meanC.y) return DOWN;
    return LEFT;
}

vector<Mat> findQR (vector<int> candidates) {
    vector<Mat> res; res.clear ();
    Mat qr, raw, gray, thres;
    qr = Mat::zeros(100, 100, CV_8UC1);
    raw = Mat::zeros(100, 100, CV_8UC1);
    gray = Mat::zeros(100, 100, CV_8UC1);
    thres = Mat::zeros(100, 100, CV_8UC1);
    int size = candidates.size ();
    if (size < 3) return res; 
    vector<Moments> mome(size);
    vector<Point2f> mean(size);
    vector<Point2f> pts1, pts2;
    // Caculate the mean point
    for (int i = 0; i < size; ++i) {
        mome[i] = moments( contours[candidates[i]], false ); 
        mean[i] = Point2f( mome[i].m10/mome[i].m00 , mome[i].m01/mome[i].m00 );
    }
    for (int A = 0; A < size; ++A) 
        for (int B = A + 1; B < size; ++B)
            for (int C = B + 1; C < size; ++C) {
                float AB = dist(mean[A],mean[B]);
                float BC = dist(mean[B],mean[C]);
                float CA = dist(mean[C],mean[A]);
                if (dist_constraint (AB, BC, CA)) continue;
                vector<int> tmp = getPoint (AB, BC, CA, A, B, C);
                int top = tmp[0]; int left = tmp[1]; int right = tmp[2];
                // Use cross product to determine left and right
                if (crossProduct (mean[left], mean[top], mean[right])) 
                    swap (left, right);
                // Draw the FIG
                drawContours (frame, contours, candidates[top] , Scalar(255,0,0), 2, 8, hierarchy, 0 );
                drawContours (frame, contours, candidates[left] , Scalar(0,255,0), 2, 8, hierarchy, 0 );
                drawContours (frame, contours, candidates[right] , Scalar(0,0,255), 2, 8, hierarchy, 0 );
                // Find all corners
                pts1 = findCorners (candidates[top], candidates[left], candidates[right], findOrientation (mean[top], mean[left], mean[right]));
                /*
                circle (frame, cvPoint (pts1[0].x, pts1[0].y), 10 , Scalar(255,0,0), 2, 8, 0);
                circle (frame, cvPoint (pts1[2].x, pts1[2].y), 10, Scalar(0,255,0), 2, 8, 0);
                circle (frame, cvPoint (pts1[1].x, pts1[1].y), 10, Scalar(0,0,255), 2, 8, 0);
                circle (frame, cvPoint (pts1[3].x, pts1[3].y), 10, Scalar(255,255,255), 2, 8, 0);
                */
                pts2.clear ();
                pts2.push_back(Point2f(0,0));
				pts2.push_back(Point2f(qr.cols,0));
				pts2.push_back(Point2f(qr.cols, qr.rows));
				pts2.push_back(Point2f(0, qr.rows));
                // Do perspective transform
                Mat M = getPerspectiveTransform (pts1,pts2);
                warpPerspective (frame, raw, M, Size (qr.cols,qr.rows));
                copyMakeBorder (raw, qr, 10, 10, 10, 10,BORDER_CONSTANT, Scalar(255,255,255));
                cvtColor (qr,gray,CV_RGB2GRAY);
                threshold (gray, thres, 127, 255, CV_THRESH_BINARY);
                res.push_back (thres);
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
