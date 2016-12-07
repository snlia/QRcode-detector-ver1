#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include "swordfeng.h"
#include <boost/program_options.hpp>

#define sqr(x) ((x) * (x))
#define INF 10000000
#define pause for (int key = -1; !~key; key = waitKey (100))

using namespace std;
using namespace cv;
namespace po = boost::program_options;

enum {UP, RIGHT, DOWN, LEFT};

vector<Vec4i> hierarchy;
vector<vector<Point>> contours;
Mat frame, rawFrame, tmpFrame;
int f [1000000];

bool useblur = 1;
bool usequad = 1;
bool useimage = 0;
int cannylow = 25;
int cannyhigh = 150;
int hierarchythre = 4;
int qrsize = 100;
double areathre = 0.3;
double distthre = 0.2;

extern void LocalThBinarization (Mat qr, Mat &out);
extern void LocalPreWorkGray (Mat &qrGray);


double dist (Point2f x, Point2f y) {
    // Return the distance between two points
    return sqrt (sqr (x.x - y.x) + sqr (x.y - y.y));
}

double crossProduct (Point2f A, Point2f B, Point2f C) {
    return ((B.x - A.x) * (C.y - B.y) - (C.x - B.x) * (B.y - A.y));
}

double distLine (Point2f X, Point2f A, Point2f B) {
    // Returen the distance between Point X and Line AB
    return abs (crossProduct (X, A, B) / dist (A, B));
}

double dotProduct (Point2f A, Point2f B) {
    return A.x * B.x + A.y * B.y;
}

double cosAngle (Point2f A, Point2f B, Point2f C, Point2f D) {
    return (dotProduct (Point2f (B.x - A.x, B.y - A.y), Point2f (D.x - C.x, D.y - C.y)) /
            (dist (A, B) * dist (C, D)));
}

// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
Point2f intersection (Point2f o1, Point2f p1, Point2f o2, Point2f p2) {
    Point2f x = o2 - o1;
    Point2f d1 = p1 - o1;
    Point2f d2 = p2 - o2;

    float cross = d1.x*d2.y - d1.y*d2.x;

    double t1 = (x.x * d2.y - x.y * d2.x)/cross;

    return o1 + d1 * t1;
}

int countHierarchy (int x) {
    // return the maxium deepth
    if (~f[x]) return f[x];
    int res = 0;
    int sx = hierarchy[x][2];
    if (~sx) {
        res = countHierarchy (sx) + 1;
        if (~hierarchy[sx][1]) res = -INF;
    }
    f[x] = res;
    return res;
}

vector<int> findCandidates (Mat frame) {
    vector<int> res, tmp;
    res.clear ();
    tmp.clear ();

    for (int i = 0; i < contours.size (); ++i) f[i] = -1;

    for (int i = 0; i < contours.size (); ++i) {
        if (usequad) {
            if (contours[i].size () == 4 && countHierarchy (i) >= hierarchythre) 
                tmp.push_back (i);
        }
        else if (countHierarchy (i) >= hierarchythre)
            tmp.push_back (i);
    }

    // the candidates cannot overlap
    for (int i = 0; i < contours.size (); ++i) f[i] = 1; 
    for (int i = 0; i < tmp.size (); ++i) {
        for (int x = hierarchy[tmp[i]][2]; ~x; x = hierarchy[x][2]) f[i] = 0;
    }
    for (int i = 0; i < tmp.size (); ++i)
        if (f[tmp[i]]) res.push_back (tmp[i]);

    return res;
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
        if (abs (BC - CA) > distthre * max (BC, CA)) return 1;
        return 0;
    }
    if (BC > AB && BC > CA) {
        if (abs (AB - CA) > distthre * max (CA, AB)) return 1;
        return 0;
    }
    if (CA > AB && CA > BC) {
        if (abs (BC - AB) > distthre * max (BC, AB)) return 1;
        return 0;
    }
    return 1;
}

bool area_constraint (double areaA, double areaB, double areaC) {
    double areaMean = (areaA + areaB + areaC) / 3;
    double sigma = sqr (areaA - areaMean) + sqr (areaB - areaMean) + sqr (areaC - areaMean);
    sigma /= 3 * sqr (areaMean);
    return sigma > areathre;
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

Point2f findN (Point2f P1, Point2f P2, Point2f P4, int top, int left, int right) {
    Point2f verP1 = findAwayFromPoint (top, P1);

    // Find the right aligned point for corner P2
    Point2f verP2;
    double maxTheta = -INF;
    for (int i = 0; i < contours[right].size (); ++i) {
        double tmp = cosAngle (P1, verP1, P2, contours[right][i]);
        if (tmp > maxTheta) {
            maxTheta = tmp;
            verP2 = contours[right][i];
        }
    }
    // Find the left aligned point for corner P4
    Point2f verP4;
    maxTheta = -INF;
    for (int i = 0; i < contours[left].size (); ++i) {
        double tmp = cosAngle (P1, verP1, P4, contours[left][i]);
        if (tmp > maxTheta) {
            maxTheta = tmp;
            verP4 = contours[left][i];
        }
    }

    // Caculate the intersection of P1verP1 and P2verP2, we use the result to do the first perspective transform
    Point2f P3, originP3;
/*    if (dist (P1, intersection (P1, verP1, P2, verP2)) > dist (P1, intersection (P1, verP1, P4, verP4)))
        originP3 = P3 = intersection (P1, verP1, P2, verP2);
    else
        originP3 = P3 = intersection (P1, verP1, P4, verP4);
        */
    originP3 = intersection (P2, verP2, P4, verP4);

    // relocate P3
    vector<Point2f> pts1, pts2;
    Mat M, gray, firstImg, bin;
    double ratio, optRatio = 0;
    int minFlag = INF;
    for (ratio = -0.2; ratio < 0.1; ratio += 0.01) {
        P3 = originP3 + (verP1 - P1) * ratio;
        pts1.clear (); pts2.clear ();
        pts1.push_back (P1); pts1.push_back (P2);
        pts1.push_back (P3); pts1.push_back (P4);
        pts2.push_back (Point2f (0, 0));
        pts2.push_back (Point2f (qrsize, 0));
        pts2.push_back (Point2f (qrsize, qrsize));
        pts2.push_back (Point2f (0, qrsize));
        M = getPerspectiveTransform (pts1, pts2);
        warpPerspective (rawFrame, firstImg, M, Size (qrsize, qrsize));
        // convert to binary image
        cvtColor (firstImg, gray,CV_RGB2GRAY);
        LocalPreWorkGray (gray);
        //    threshold (gray, bin, 180, 255, CV_THRESH_BINARY);
        LocalThBinarization (gray, bin);
        int flag = 0;
        for (int i = int (qrsize * 0.9); i < qrsize; ++i) {
            flag += bin.at<uchar>(qrsize - 1, i) == 0;
            flag += bin.at<uchar>(i, qrsize - 1) == 0;
        }
        /*
        if (minFlag == INF) {
            imshow ("bin", bin);
            pause;
        }*/
        if (flag <= minFlag) {
            minFlag = flag;
            optRatio = ratio;
        }
   //     printf ("%d\n", flag);
  //      printf ("%.4lf\n", ratio);
    }
//    printf ("OPT %.4lf\n", optRatio);

    // Caculate the optium P3
    P3 = originP3 + (verP1 - P1) * optRatio;
    pts1.clear (); pts2.clear ();
    pts1.push_back (P1); pts1.push_back (P2);
    pts1.push_back (P3); pts1.push_back (P4);
    pts2.push_back (Point2f (0, 0));
    pts2.push_back (Point2f (qrsize, 0));
    pts2.push_back (Point2f (qrsize, qrsize));
    pts2.push_back (Point2f (0, qrsize));
    M = getPerspectiveTransform (pts1, pts2);
    warpPerspective (rawFrame, firstImg, M, Size (qrsize, qrsize));
    // convert to binary image
    cvtColor (firstImg, gray,CV_RGB2GRAY);
    LocalPreWorkGray (gray);
    //    threshold (gray, bin, 180, 255, CV_THRESH_BINARY);
    LocalThBinarization (gray, bin);

    //imshow ("bin", bin);
    //pause;

    // Find K1, K2
    vector<Point2f> K1, K2;
    K1.push_back (Point2f(qrsize - 1, qrsize - 1));
    K2.push_back (Point2f(qrsize - 1, qrsize - 1));
    double minK1 = INF, minK2 = INF;
    for (int i = qrsize - 1; i > int (0.7 * qrsize); --i) 
        for (int j = qrsize - 1; j > int (0.7 * qrsize); --j)
            if (bin.at<uchar>(j, i) == 0) {
                // Update K1
                if (minK1 > (qrsize - i + 0.0) / j) {
                    minK1 = (qrsize - i + 0.0) / j;
                    K1[0] = Point2f(i, j);
                }
                // Update K2
                if (minK2 > (qrsize - j + 0.0) / i) {
                    minK2 = (qrsize - j + 0.0) / i;
                    K2[0] = Point2f(i, j);
                }
            }
    // P3 should be the intersection of P4K2 and P2K1
    M = getPerspectiveTransform (pts2, pts1);
    vector<Point2f> realK1, realK2;
    perspectiveTransform (K1, realK1, M);
    perspectiveTransform (K2, realK2, M);
    return intersection (P4, realK2[0], P2, realK1[0]);
    puts ("Next!");
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
    res.push_back (findN (A, B, C, top, left, right));
    return res;
}

vector<Mat> findQR (vector<int> candidates) {
    vector<Mat> res; res.clear ();
    Mat qr, raw;
    qr = Mat::zeros(qrsize, qrsize, CV_8UC1);
    raw = Mat::zeros(qrsize, qrsize, CV_8UC1);
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
                pts2.push_back(Point2f(qrsize,0));
                pts2.push_back(Point2f(0, qrsize));
                pts2.push_back(Point2f(qrsize, qrsize));
                // Do perspective transform
                Mat M = getPerspectiveTransform (pts1,pts2);
                warpPerspective (rawFrame, raw, M, Size (qr.cols,qr.rows));
                copyMakeBorder (raw, qr, 10, 10, 10, 10, BORDER_CONSTANT, Scalar(255,255,255));

                // Draw the FIP
                drawContours (frame, contours, candidates[top] , Scalar(255,0,0), 2, 8, hierarchy, 0 );
                drawContours (frame, contours, candidates[left] , Scalar(0,255,0), 2, 8, hierarchy, 0 );
                drawContours (frame, contours, candidates[right] , Scalar(0,0,255), 2, 8, hierarchy, 0 );

                res.push_back (qr);
                return res;
            }
    return res;
}

int parameter_init (int argc, const char *argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "show this message.")
        ("image", "use image detection.")
        ("noblur", "do not use blur.")
        ("noquad", "do not use quadrangle detection.")
        ("clow", po::value<double>(), "set up the low thresold of canny, default 75.")
        ("chigh", po::value<double>(), "set up the high thresold of canny, default 200.")
        ("size", po::value<int>(), "set up the qr code size, default 100.")
        ("hthre", po::value<int>(), "set up the thresold of hierarchy, default 5.")
        ("athre", po::value<double>(), "set up the thresold of area constraint, default 0.2.")
        ("dthre", po::value<double>(), "set up the thresold of distance constraint, default 0.005.")
        ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    

    if (vm.count("help")) {
        cout << desc << endl;
        return EOF;
    }
    if (vm.count ("image")) {
        cout << "Image detection." << endl;
        useimage = 1;
    }
    if (vm.count ("noblur")) {
        cout << "Do not use blur." << endl;
        useblur = 0;
    } 
    if (vm.count ("noquad")) {
        cout << "Do not use quadrangle detection." << endl;
        usequad = 0;
    }
    if (vm.count ("clow")) {
        cannylow = vm["clow"].as<double> ();
        printf ("canny low thresold is set as %lf\n", cannylow);
    }
    if (vm.count ("chigh")) {
        cannyhigh = vm["chigh"].as<double> ();
        printf ("canny high thresold is set as %lf\n", cannyhigh);
    }
    if (vm.count ("size")) {
        qrsize = vm["size"].as<int> ();
        printf ("QRcode size is set as %d\n", qrsize);
    }
    if (vm.count ("hthre")) {
        hierarchythre = vm["hthre"].as<int> ();
        printf ("hierarchy constraint thresold is set as %d\n", hierarchythre);
    }
    if (vm.count ("athre")) {
        areathre = vm["athre"].as<double> ();
        printf ("area constraint thresold is set as %lf\n", areathre);
    }
    if (vm.count ("dthre")) {
        distthre = vm["dthre"].as<double> ();
        printf ("distance constraint thresold is set as %lf\n", distthre);
    }
    return 0;
}

int main(int argc, const char *argv[]) {

    if (parameter_init (argc, argv)) return 0;

    Mat gray(frame.size(), CV_MAKETYPE(frame.depth(), 1));
    Mat detected_edges(frame.size(), CV_MAKETYPE(frame.depth(), 1));
    Mat edges (frame.size (), CV_MAKETYPE (frame.depth (), 1));
    tmpFrame = Mat::zeros(qrsize, qrsize, CV_8UC1);

    if (useimage) {
        cout << "Plz input the filename:" << endl;
        string filename;
        cin >> filename;
        for (; filename != "q"; cin >> filename) {
            cout << filename << endl;
            frame = imread ("../data/" + filename);
            frame.copyTo (rawFrame);
            // Change to grayscale
            cvtColor (frame, gray, CV_RGB2GRAY);

            if (useblur) {
                blur (gray, detected_edges, Size(3,3));
                // Find FIP candidates
                Canny (detected_edges, edges, cannylow, cannyhigh, 3);		// Apply Canny edge detection on the gray image
            }
            else 
                Canny (gray, edges, cannylow, cannyhigh, 3);		// Apply Canny edge detection on the gray image
            if (usequad) {
                contours.clear ();
                vector<Point> approx;
                vector<vector<Point>> raw_contours;
                findContours (edges, raw_contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
                for (int i = 0; i < raw_contours.size (); ++i) {
                    approxPolyDP (Mat(raw_contours[i]), approx, arcLength (Mat(raw_contours[i]), true) * 0.02, true);
                    contours.push_back (approx);
                }
            }
            else 
                findContours (edges, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
            vector<int> candidates = findCandidates (gray);

            vector<Mat> qrs = findQR (candidates);

            for (int i = 0; i < qrs.size (); ++i) {
                String Name = "QR";
                String id;
                stringstream sout;
                sout << i;
                id = sout.str ();
                Name += id;
                imshow (Name, qrs[i]);
            }

            imshow ("Image", frame);

            cout << "Press any key to continue." << endl;
            pause;
            cout << "Plz input the filename:" << endl;
        }
    }
    else {
        VideoCapture capture(0);

        capture >> frame;

        cout << "Press any key to return." << endl;

        for (int key = -1; !~key; capture >> frame) {
            frame.copyTo (rawFrame);
            // Change to grayscale
            cvtColor (frame, gray, CV_RGB2GRAY);

            if (useblur)
                blur (gray, detected_edges, Size(3,3));
            // Find FIP candidates
            Canny (detected_edges, edges, cannylow, cannyhigh, 3);		// Apply Canny edge detection on the gray image
            if (usequad) {
                contours.clear ();
                vector<Point> approx;
                vector<vector<Point>> raw_contours;
                findContours (edges, raw_contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
                for (int i = 0; i < raw_contours.size (); ++i) {
                    approxPolyDP (Mat(raw_contours[i]), approx, arcLength (Mat(raw_contours[i]), true) * 0.02, true);
                    contours.push_back (approx);
                }
            }
            else 
                findContours (edges, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
            vector<int> candidates = findCandidates (gray);

            vector<Mat> qrs = findQR (candidates);

            for (int i = 0; i < qrs.size (); ++i) {
                String Name = "QR";
                String id;
                stringstream sout;
                sout << i;
                id = sout.str ();
                Name += id;
                imshow (Name, qrs[i]);
            }

            imshow ("Image", frame);

            key = waitKey (100);
        }
    }

    return 0;
}
