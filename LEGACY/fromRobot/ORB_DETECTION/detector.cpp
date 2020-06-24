#include <opencv2/opencv.hpp>
#include "eyebot++.h"
#include <Python.h>
#include <string>
#include <chrono>
#include <stdio.h>
#include <vector>
#include <algorithm> // for max and min
#include <cmath>

#include <iostream>
#include <fstream>

//#include <numpy/arrayobject.h>

#include "TSDR.cpp"

#define RESOLUTION QVGA
#define SIZE QVGA_SIZE
#define PIXELS QVGA_PIXELS
#define WIDTH QVGA_X
#define HEIGHT QVGA_Y

#define MAX_LABELS 20

#define FREQ1 15 /* in Hz */

using namespace cv;
using namespace std;
using namespace chrono;
using namespace TSDR;

BYTE rgb[SIZE];
vector <string> labels;

PyObject *pModule, *pFunc;
//PyObject *pArgs, *pValue1, *pValue2;

char sessionID[15];
int num = 0;

bool my_compare (ROI a, ROI b)
{
    return a.priority > b.priority;
}

void generateSessionID()
{   time_t rawtime;
    struct tm * timeinfo;
    time (&rawtime);
    timeinfo = localtime (&rawtime);
    strftime(sessionID, 15, "%G%m%d%H%M%S", timeinfo);
}

void detectROIs(list<ROI>& rois) {
    
    //static Mat grey;
    static vector<KeyPoint> keypoints;
    static vector<int> keypoint_indexes;
    static Mat image = cv::Mat(HEIGHT,WIDTH, CV_8UC3, rgb);
    //cv::cvtColor(image, grey, CV_RGB2GRAY);
    
    
    
    //static Ptr<FeatureDetector> detector = FeatureDetector::create("ORB");
    static Ptr<ORB> detector = ORB::create();
    
    //detector->detect(image, keypoints, Mat());
    
    static Mat mask = Mat::zeros(HEIGHT,WIDTH,CV_8U);
    mask(Rect(0,0,WIDTH,HEIGHT*0.8))=1;
    
    detector->detect(image, keypoints, mask);
    
    
    Mat image_keypoints = image.clone();
    
    //drawKeypoints(image, keypoints, image,Scalar(255,153,255), DrawMatchesFlags::DEFAULT );
    drawKeypoints(image, keypoints, image_keypoints,Scalar(255,153,255), DrawMatchesFlags::DEFAULT );
    
    vector<Point2f> points;
    KeyPoint::convert(keypoints, points, keypoint_indexes);
    
    static const int WINDOW_SIZE = 90;//32;
    static const int THRESHOLD = 5;//24;
    static const int STEP_SIZE = 16;
    
    // this can be optimised further
    int i,j;
    for(i=0; i<WIDTH-WINDOW_SIZE;i=i+STEP_SIZE) {
        for(j=0; j<HEIGHT-WINDOW_SIZE; j=j+STEP_SIZE) {
            int density = 0;
            
            //use iterator here instead
            for (Point2f p : points) {
                int x = (int) p.x;
                int y = (int) p.y;
                if( (i<x&&x<i+WINDOW_SIZE)&&(j<y&&y<j+WINDOW_SIZE)) density++;
            }
            if(density>THRESHOLD ) {
                ROI temp = ROI(i,j,WINDOW_SIZE,WINDOW_SIZE,density);
                rois.push_front(temp);
            }
            
        }
    }
    
    rois.sort(my_compare);
    
    KeyPoint::convert(keypoints, points, keypoint_indexes);
    
    

    LCDImage(rgb);
    
    
    
    /*
    std::vector<uchar> tmp_array(SIZE);
    tmp_array.assign(image_keypoints.data, image_keypoints.data + 3*image_keypoints.total());
    BYTE* img = &tmp_array[0];
    
    LCDImage(img);
    */
    
    return;
}

void showROIs(list<ROI>& rois) {
    list<ROI>::iterator it;
    for (it = rois.begin(); it != rois.end(); it++) {
        int x = it->x;
        int y = it->y;
        int xs = it->xs;
        int ys = it->ys;
        LCDArea(x,y,x+xs,y+ys,GREEN,0);
    }
    return;
}

void TSD_reporting(float td) {
    //printf("FPS: %f\n", 1/td);
    
    LCDSetPrintf(0,52, "FPS (actual): %f\n", 1/td);
    LCDSetPrintf(1,52, "FPS (limit): %d\n", FREQ1);
    
}

void NMSuppression(list<ROI>& rois) {
    /*
     1. Get highest priority ROI
     2. Remove all ROIs with high overlap to highest priority ROI
     3. Repeat with next highest ROI until reach the end of the list
     note: list is already sorted by priority
     */
    
    static const float OVERLAP_THRESHOLD = 0.10; // was 0.05
    list<ROI>::iterator it1;
    for (it1 = rois.begin(); it1 != rois.end(); it1++) {
        list<ROI>::iterator it2;
        
        int area = (it1->xs)*(it1->ys);
        for (it2 = next(it1); it2 != rois.end(); it2++) {
            int x_overlap = max(0, min(it1->x+it1->xs, it2->x+it2->xs) - max(it1->x, it2->x));
            int y_overlap = max(0, min(it1->y+it1->ys, it2->y+it2->ys) - max(it1->y, it2->y));
            int overlapArea = x_overlap * y_overlap;
            
            if(overlapArea > OVERLAP_THRESHOLD*area) {
                it2 = rois.erase(it2);
                it2--;
            }
        }
    }
    return;
}

void trackDetections(list<ROI>& rois, list<Detection>& detections) {
    /*
        compare with existing detections and delete ROIs that have been classified before?
     */
    
    const static int DISTANCE_THRESHOLD = 35;
    list<Detection>::iterator it1;
    for (it1=detections.begin(); it1!=detections.end();it1++) {
        list<ROI>::iterator it2;
        for(it2=rois.begin();it2!=rois.end();it2++) {
            int it1_xc = it1->x + it1->xs/2;
            int it1_yc = it1->y + it1->ys/2;
            int it2_xc = it2->x + it2->xs/2;
            int it2_yc = it2->y + it2->ys/2;
            
            int dx = abs(it1_xc-it2_xc);
            int dy = abs(it1_yc-it2_yc);
            
            int dist = sqrt(dx*dx+dy*dy);
            
            //printf("dist = %d\n", dist);
            
            if(dist < DISTANCE_THRESHOLD) {
                // update location of detection
                
                const static int SIZE_OF_CLASSIFIER = 128;
                /*
                if(it2_xc-SIZE_OF_CLASSIFIER/2<0) it1.update_horizontals(0,SIZE_OF_CLASSIFIER);
                else if(x_c+SIZE_OF_CLASSIFIER/2>WIDTH) x_c = WIDTH-SIZE_OF_CLASSIFIER/2;
                if(y_c-SIZE_OF_CLASSIFIER/2<0) y_c = SIZE_OF_CLASSIFIER/2;
                else if(y_c+SIZE_OF_CLASSIFIER/2>HEIGHT) y_c = HEIGHT-SIZE_OF_CLASSIFIER/2;
                 */
                if(it2_xc-SIZE_OF_CLASSIFIER/2<0) {
                    it1->update_horizontals(0,SIZE_OF_CLASSIFIER);
                } else if(it2_xc+SIZE_OF_CLASSIFIER/2>WIDTH) {
                    it1->update_horizontals(WIDTH-SIZE_OF_CLASSIFIER, SIZE_OF_CLASSIFIER);
                } else {
                    it1->update_horizontals(it2_xc-SIZE_OF_CLASSIFIER/2,SIZE_OF_CLASSIFIER);
                }
                
                if(it2_yc-SIZE_OF_CLASSIFIER/2<0) {
                    it1->update_verticals(0,SIZE_OF_CLASSIFIER);
                } else if(it2_yc+SIZE_OF_CLASSIFIER/2>HEIGHT) {
                    it1->update_verticals(HEIGHT-SIZE_OF_CLASSIFIER, SIZE_OF_CLASSIFIER);
                } else {
                    it1->update_verticals(it2_yc-SIZE_OF_CLASSIFIER/2,SIZE_OF_CLASSIFIER);
                }

                
                rois.erase(it2);
                break;
            }
        }
        
        // if it has reached this point, it cannot be tracked
        if(it2==rois.end()) {
            it1 = detections.erase(it1);
            it1--;
        }
        
    }
    return;
}

void show_detections(list<Detection>& detections) {
    list<Detection>::iterator it1;
    for (it1=detections.begin(); it1!=detections.end();it1++) {
        if(it1->object_class==0) continue;
        
        const static float X_M= 0.18;
        const static float X_C= -2.3;
        const static float Y_M= 0.06;
        const static float Y_C= 1.4;
        

        LCDArea(it1->x,it1->y,it1->x+it1->xs-1,it1->y+it1->ys-1,YELLOW,0);
        LCDSetPos(Y_M*it1->y+Y_C,X_M*it1->x+X_C);
        LCDPrintf("%s:%f",labels[it1->object_class].c_str(),it1->confidence);
    }
    
    return;
}


void classification(list<ROI>& rois, list<Detection>& detections) {
    
    int index;
    double confidence;
    
    if(rois.empty()) return;
    ROI *roi = &rois.front();
    int x_c = roi->x + (roi->xs)/2;
    int y_c = roi->y + (roi->ys)/2;
    
    const static int SIZE_OF_CLASSIFIER = 128;
    x_c = min(WIDTH-SIZE_OF_CLASSIFIER/2,max(SIZE_OF_CLASSIFIER/2,x_c));
    y_c = min(HEIGHT-SIZE_OF_CLASSIFIER/2,max(SIZE_OF_CLASSIFIER/2,y_c));
    
    LCDArea(x_c-SIZE_OF_CLASSIFIER/2,y_c-SIZE_OF_CLASSIFIER/2,x_c+SIZE_OF_CLASSIFIER/2-1,y_c+SIZE_OF_CLASSIFIER/2-1,RED,0);
    
    
    Mat image = cv::Mat(HEIGHT,WIDTH, CV_8UC3, rgb);
    //Rect highest_priority(roi->x,roi->y,roi->xs,roi->ys);
    //Rect highest_priority(x_c - SIZE_OF_CLASSIFIER/2,y_c - SIZE_OF_CLASSIFIER/2 ,SIZE_OF_CLASSIFIER,SIZE_OF_CLASSIFIER);
    Rect highest_priority(x_c - SIZE_OF_CLASSIFIER/2,y_c - SIZE_OF_CLASSIFIER/2 ,SIZE_OF_CLASSIFIER,SIZE_OF_CLASSIFIER);
    cvtColor(image, image,CV_RGB2BGR);
    Mat tmp = image(highest_priority);
    Mat cropped;
    tmp.copyTo(cropped); // not using pointers
    
    
    
    
    
    //image.copyTo(cropped,highest_priority);

    /*
    namedWindow( "Display window", WINDOW_AUTOSIZE );
    imshow( "Display window", cropped );
    waitKey(0);
     */
    
    PyObject *pArgs = PyTuple_New(3);
    char *img_data = (char *)cropped.data;
    int len = SIZE_OF_CLASSIFIER*SIZE_OF_CLASSIFIER*3;
    PyObject *pValue1 = PyMemoryView_FromMemory(img_data,len,PyBUF_READ);
    if (!pValue1) {
        //printf("1\n");
        Py_XDECREF(pValue1);
        Py_DECREF(pArgs);
        Py_DECREF(pModule);
        fprintf(stderr, "Cannot convert argument\n");
        return;
    }
    
    
    //printf("2\n");
    PyTuple_SetItem(pArgs, 0, pValue1);
    PyTuple_SetItem(pArgs, 1, PyLong_FromLong(128)); // set height
    PyTuple_SetItem(pArgs, 2, PyLong_FromLong(128)); // set width
    //printf("2.1\n");
    
    PyObject *pValue2 = PyObject_CallObject(pFunc, pArgs);
    //printf("here1\n");
    //printf("2.2\n");
    //Py_DECREF(pValue1);
    //Py_DECREF(pArgs);
    //printf("2.3\n");
    if (pValue2 != NULL) {
        //printf("3\n");
        //PyObject *pResult1 = PyTuple_GetItem(pValue2,0);
        //index = (uint) PyLong_AsLong(pResult1);
        //index = int (PyLong_AsLong(pResult1));
        index = int (PyLong_AsLong(PyTuple_GetItem(pValue2,0)));
        //printf("index=%i",index);
        //PyObject *pResult2 = PyTuple_GetItem(pValue2,1);
        //confidence = PyFloat_AsDouble(pResult2);
        confidence =PyFloat_AsDouble(PyTuple_GetItem(pValue2,1));
        
        //printf("index = %i, confidence = %f\n", index, confidence);
        
        
        // fault is here
        //Py_XDECREF(pResult1);
        //Py_XDECREF(pResult2);
        Py_XDECREF(pValue2);
    }
    else {
        //printf("4\n");
        Py_XDECREF(pValue2);
        PyErr_Print();
        fprintf(stderr,"Call failed\n");
        return;
    }
    
    
    
    //printf("5\n");
    
    Detection observed(index, confidence,x_c-SIZE_OF_CLASSIFIER/2,y_c-SIZE_OF_CLASSIFIER/2,SIZE_OF_CLASSIFIER,SIZE_OF_CLASSIFIER );
    
    //printf("6\n");
    
    detections.push_back(observed);
    //printf("\nlabel: %s, confidence: %f\n", labels[index].c_str(), confidence);
    
    
    int train = 0;
    if(train==1) {
        char filename [50];
        snprintf(filename,50,"images/%s/%s_%04d.jpg", labels[index].c_str(), sessionID, num);
        std::string str(filename);
        //str = "crossing/" + EXTENSION;
        num++;
        
        imwrite(filename,cropped);
        
    }
    
    return;
}

void read_labels() {
    ifstream f;
    f.open("tf_files/labels.txt");
    while(!f.eof()) {
        string s;
        getline(f,s);
        labels.push_back(s);
        //printf("%s\n", s.c_str());
    }
    f.close();
    return;
}

void traffic_sign_detection() {
    // to monitor CPU consumption
    static high_resolution_clock::time_point t1;
    t1 = high_resolution_clock::now();
    
    list<ROI> _ROIs;
    static list<Detection> detections;
    
    detectROIs(_ROIs);
    
    NMSuppression(_ROIs);
    //showROIs(_ROIs);
    trackDetections(_ROIs, detections);
    showROIs(_ROIs);
    classification(_ROIs, detections);
    
    show_detections(detections);
    
    //LCDImage(rgb);
    
    
    /*
     1. detect ROIs (input: rgb image; output: list of ROIs
     2. supress non-maximal ROIs
     3. track existant detections
     4. choose highest-priority ROI to classify
     5. perform classification and return new detection
     6. display detections
     */
    
    //printf("\nHello, World!\n");
    
    static high_resolution_clock::time_point t2;
    t2 = high_resolution_clock::now();
    static float td;
    td = (duration_cast<duration<double> >(t2-t1)).count();
    
    
    //printf("number of detections: %d\n", detections.size());
    
    TSD_reporting(td);
}

int main ()
{
    LCDSetPrintf(0,0, "[PLEASE WAIT]");
    char *app_name = (char *)"EyeBot Traffic Sign Recognition and Detection";
    Py_SetProgramName((wchar_t*)app_name);
    Py_Initialize();
    
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\"./scripts/\")");
    
    PyObject *pName = PyUnicode_FromString("label");
    pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    
    if(pModule==NULL) {
        PyErr_Print();
        Py_DECREF(pModule);
        return -1;
    }
    
    
    pFunc = PyObject_GetAttrString(pModule, "label");
    if(!pFunc || !PyCallable_Check(pFunc)) {
        if (PyErr_Occurred()) PyErr_Print();
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
        return -1;
    }
    
    read_labels();
    //generateSessionID();
    
    printf("ALL GOOD AND LOADED! \n\n\n\n\n");
    
    
    CAMInit(RESOLUTION);
    LCDMenu("","","","END");
    
    while(1) {
        static high_resolution_clock::time_point tc;
        tc = high_resolution_clock::now();
        
        CAMGet(rgb);
        
        
        /* non-interupt timer 1 */
        static high_resolution_clock::time_point t1 = high_resolution_clock::now();
        static float td_1;
        td_1 = (duration_cast<duration<double> >(tc-t1)).count();
        if( 1000/FREQ1 < 1000*td_1) {
            traffic_sign_detection();
            t1 = high_resolution_clock::now();
        }
        
        /* io */
        static int key;
        key = KEYRead();
        if(key == KEY4) break;
        
        OSWait(100);
    }
    
    //Py_XDECREF(pFunc);
    //Py_DECREF(pModule);
    
    /*
    if (Py_FinalizeEx() < 0) {
        return 1;
    }
    */
    
    CAMRelease();
    return 0;
}
