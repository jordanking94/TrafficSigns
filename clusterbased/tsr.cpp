#include <Python.h>
#include <opencv2/opencv.hpp>
#include "eyebot++.h"
#include "TSDR.hpp"

#include <cstdio>
#include <string>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;
using namespace chrono;
using namespace TSDR;

#define RESOLUTION QVGA
#define SIZE QVGA_SIZE
#define PIXELS QVGA_PIXELS
#define WIDTH QVGA_X
#define HEIGHT QVGA_Y


#define MAX_LABELS 20
#define FREQ1 25//15
#define NO_HUE 255

#define FOREGROUND 1
#define BACKGROUND 0

#define ELEMENT_SHAPE MORPH_ELLIPSE//MORPH_RECT
#define EROSION_SIZE 1
#define EROSION_ITERATIONS 1
#define DILATION_SIZE 3//6
#define DILATION_ITERATIONS 2//1

#define ROI_MIN_PERMISSIBLE_AREA 50
#define ROI_FROM_BLOB_SIZE 0

#define TRACKING_DISTANCE_THRESHOLD 50//35

#define MAX_CLASSIFICATIONS_PER_FRAME 1
#define KEYPOINT_RADII 5



#define DETECTION_THRESHOLD 5 // number of keypoints in cluster  before it is recorded as an ROI

void Program_Initialisation();
int Python_Initialisation();
void generateSessionID();
void read_labels();
void traffic_sign_detection();
void RGB2HSI(BYTE* rgb, BYTE* hsi);
void HSI2BIN(BYTE* hsi, BYTE* bin);
void Dilation(Mat *InputArray, Mat *OutputArray);
void Erosion(Mat *InputArray, Mat *OutputArray);
void showROIs(list<ROI>* ROIs);
void detectClusters(vector<Cluster>* clusters);
void detectROIs(list<ROI>* ROIs, vector<Cluster*>* clusters);
void show_detections(list<Detection_Profile>* _profiles);
void trackDetections(list<ROI>* _ROIs, list<Detection_Profile>* _profiles);
void classification(list<ROI>* _ROIs, list<Detection_Profile>* _profiles);
void conclude_frame(list<Detection_Profile>* _profiles);

BYTE rgb[SIZE];
BYTE hsi[SIZE];
BYTE bin[PIXELS];
BYTE mask[PIXELS];

Mat rgbMat;
Mat hsiMat;
Mat binMat;
Mat maskMat;

list <Detection> detections;
list <Detection_Profile> profiles; // should be list<Detection_Profile*> profiles instead!

vector <string> labels;

Mat cc_labelImage;
Mat cc_stats;
Mat cc_centroids;

PyObject *pModule, *pFunc;

char sessionID[15];
int num = 0;

int colourmap[] = {LIGHTGRAY, CYAN, MAGENTA, ORANGE, RED, BLUE, PURPLE,
                    MAROON, YELLOW, TEAL, NAVY, OLIVE, GREEN,
                    SILVER, GRAY, DARKGRAY};

int main() {
    LCDSetPrintf(0,0, "[PLEASE WAIT]");
    
    Program_Initialisation();
    int py_error = Python_Initialisation();
    if(py_error!=0) return py_error;
    
    LCDMenu("CAMERA","DASHBOARD","HEADLESS","END");
    
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
    }
    
    CAMRelease();
    
    Py_XDECREF(pFunc);
    Py_DECREF(pModule);
    
    if (Py_FinalizeEx() < 0) {
        return 1;
    }
    
    return 0;
}

void Program_Initialisation() {
    read_labels();
    generateSessionID();
    CAMInit(RESOLUTION);

    binMat = Mat(HEIGHT,WIDTH, CV_8UC1, bin);
    maskMat = Mat(HEIGHT,WIDTH, CV_8UC1, mask);
    rgbMat = Mat(HEIGHT, WIDTH, CV_8UC3, rgb);
}

int Python_Initialisation() {
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
    
    return 0;
}


void generateSessionID()
{   time_t rawtime;
    struct tm * timeinfo;
    time (&rawtime);
    timeinfo = localtime (&rawtime);
    strftime(sessionID, 15, "%G%m%d%H%M%S", timeinfo);
}

void read_labels() {
    ifstream f;
    f.open("tf_files/labels.txt");
    while(!f.eof()) {
        string s;
        getline(f,s);
        labels.push_back(s);
    }
    f.close();
    return;
}

void Dilation(Mat *InputArray, Mat *OutputArray) {
    static Mat element = getStructuringElement( ELEMENT_SHAPE,
                                        Size( 2*DILATION_SIZE + 1, 2*DILATION_SIZE + 1 ),
                                        Point( DILATION_SIZE, DILATION_SIZE ) );
    dilate(*InputArray, *OutputArray, element, Point(-1,-1), DILATION_ITERATIONS);
}

void Erosion(Mat *InputArray, Mat *OutputArray) {
    static Mat element = getStructuringElement( ELEMENT_SHAPE,
                                               Size( 2*EROSION_SIZE + 1, 2*EROSION_SIZE + 1 ),
                                               Point( EROSION_SIZE, EROSION_SIZE ) );
    erode(*InputArray, *OutputArray, element, Point(-1,-1), EROSION_ITERATIONS);
}

void showROIs(list<ROI>* ROIs) {
    list<ROI>::iterator it;
    for (it = ROIs->begin(); it != ROIs->end(); it++) {
        int x = it->x;
        int y = it->y;
        int xs = it->xs;
        int ys = it->ys;
        LCDArea(x,y,x+xs,y+ys,GREEN,0);
    }
    return;
}

void trackDetections(list<ROI>* ROIs, list<Detection>* detections) {
    list<Detection>::iterator it1;
    for (it1=detections->begin(); it1!=detections->end();it1++) {
        list<ROI>::iterator it2;
        for(it2=ROIs->begin();it2!=ROIs->end();it2++) {
            int it1_xc = it1->x + it1->xs/2;
            int it1_yc = it1->y + it1->ys/2;
            int it2_xc = it2->x + it2->xs/2;
            int it2_yc = it2->y + it2->ys/2;
            
            int dx = it1_xc-it2_xc;
            int dy = it1_yc-it2_yc;
            
            int dist = sqrt(dx*dx+dy*dy);
            if(dist < TRACKING_DISTANCE_THRESHOLD) {
                
                it1->x = it2->x;
                it1->y = it2->y;
                it1->xs = it2->xs;
                it1->ys = it2->ys;

                ROIs->erase(it2);
                break;
            }
        }
        // if it has reached this point, it cannot be tracked
        if( it2==ROIs->end() ) {
            it1 = detections->erase(it1);
            it1--;
        }
    }
    return;
    
}

void trackDetections(list<ROI>* _ROIs, list<Detection_Profile>* _profiles) {
    list<Detection_Profile>::iterator it1;
    for (it1=_profiles->begin(); it1!=_profiles->end();it1++) {
        list<ROI>::iterator it2;
        for(it2=_ROIs->begin();it2!=_ROIs->end();it2++) {
            int it1_xc = it1->x + it1->xs/2;
            int it1_yc = it1->y + it1->ys/2;
            int it2_xc = it2->x + it2->xs/2;
            int it2_yc = it2->y + it2->ys/2;
            
            int dx = it1_xc-it2_xc;
            int dy = it1_yc-it2_yc;
            
            int dist = sqrt(dx*dx+dy*dy);
            if(dist < TRACKING_DISTANCE_THRESHOLD) {
                
                it1->x = it2->x;
                it1->y = it2->y;
                it1->xs = it2->xs;
                it1->ys = it2->ys;
                
                _ROIs->erase(it2);
                
                break;
            }
        }
        // if it has reached this point, it cannot be tracked
        if( it2==_ROIs->end() ) {
            it1 = _profiles->erase(it1);
            it1--;
        }
    }
    
    return;
}



void detectClusters(vector<Cluster>* clusters) {
    // note: should be using pointers instead!
    RGB2HSI(rgb, hsi);
    HSI2BIN(hsi,bin);
    Erosion(&binMat, &maskMat);
    Dilation(&maskMat, &maskMat);
    
    int nLabels = connectedComponentsWithStats(maskMat, cc_labelImage, cc_stats, cc_centroids, 8);
    
    for(int i = 1; i<nLabels; i++) {
        int x = cc_stats.at<int>(Point(0, i));
        int y = cc_stats.at<int>(Point(1, i));
        int w = cc_stats.at<int>(Point(2, i));
        int h = cc_stats.at<int>(Point(3, i));
        Cluster* c = new Cluster(i,x,y,w,h);
        clusters->push_back(*c);
    }
    
    static vector<KeyPoint> keypoints;
    static vector<int> keypoint_indexes;
    static Mat image = Mat(HEIGHT,WIDTH, CV_8UC3, rgb);
    
    //static Ptr<ORB> detector = ORB::create();
    static Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
    
    static Mat t_mask = Mat::zeros(HEIGHT,WIDTH,CV_8U);
    t_mask(Rect(0,0,WIDTH,HEIGHT)) = 1;
    detector->detect(image, keypoints, t_mask);
    
    // to display keypoints
    /*
     vector<KeyPoint>::iterator it;
     for(it=keypoints.begin(); it!=keypoints.end(); ++it) {
     int x = (int) it->pt.x;
     int y = (int) it->pt.y;
     int label = cc_labelImage.at<int>(y, x);
     LCDCircle(x, y, KEYPOINT_RADII, colourmap[label], 0);
     }
     */
    vector<KeyPoint>::iterator it3;
    for(it3=keypoints.begin(); it3!=keypoints.end(); ++it3) {
        int x = (int) it3->pt.x;
        int y = (int) it3->pt.y;
        int label = cc_labelImage.at<int>(y, x);
        if(label!=0) {
            clusters->at(label-1).add_keypoint(*it3);
        }
    }
    
}

void detectROIs(list<ROI> *ROIs, vector<Cluster> *clusters) {
    for(int i=0; i<int(clusters->size()); i++) {
        if(clusters->at(i).n >= DETECTION_THRESHOLD) {
            int x,y,xs,ys;
            
            if(ROI_FROM_BLOB_SIZE) {
                x = clusters->at(i).x;
                y = clusters->at(i).y;
                xs = clusters->at(i).w;
                ys = clusters->at(i).h;
            }
            else {
                x = clusters->at(i).kp_xmin;
                y = clusters->at(i).kp_ymin;
                xs = clusters->at(i).kp_xmax - clusters->at(i).kp_xmin;
                ys = clusters->at(i).kp_ymax - clusters->at(i).kp_ymin;
            }
            int area = xs*ys;
            if(area<ROI_MIN_PERMISSIBLE_AREA) continue;
            
            int priority = clusters->at(i).n-1;
            
            ROI r = ROI(x,y,xs,ys,priority, &(clusters->at(i)) );
            ROIs->push_back(r);
        }
    }

    // sort ROIs
    ROIs->sort(compareROIs);
    return;
}

void traffic_sign_detection() {
    /*
     1. detect ROIs (input: rgb image; output: list of ROIs
     2. perform non-maxima suppression
     3. track existant detections
     4. choose highest-priority ROI to classify
     5. perform classification and return new detection
     6. display detections
     */
    LCDImageStart(0,0,WIDTH,HEIGHT);
    LCDImage(rgb);
    //LCDImageStart(0,0,WIDTH,HEIGHT);
    //LCDImageBinary(mask);
    
    static high_resolution_clock::time_point t1;
    t1 = high_resolution_clock::now();
    
    vector<Cluster> clusters;
    list<ROI> ROIs;
    detectClusters(&clusters);
    detectROIs(&ROIs, &clusters);
    trackDetections(&ROIs, &profiles);
    showROIs(&ROIs);
    classification(&ROIs, &profiles);
    show_detections(&profiles);
    conclude_frame(&profiles);

    static high_resolution_clock::time_point t2;
    t2 = high_resolution_clock::now();
    static float td;
    td = (duration_cast<duration<double> >(t2-t1)).count();
    
    printf("td: %f\n", td);
    return;
}

void RGB2HSI(BYTE* rgb, BYTE* hsi)
{   int i;
    for(i=0; i<PIXELS; i++)
    {   int r,g,b,max,min,delta;
        BYTE hue = 0;
        r=rgb[3*i];
        g=rgb[3*i+1];
        b=rgb[3*i+2];
        
        max = MAX(r,MAX(g,b));
        min = MIN(r,MIN(g,b));
        delta = max - min;
        
        if (2*delta <= max) hue = NO_HUE;
        else
        {
            if (r==max) hue = 42 + 42*int(double(g-b)/double(delta));
            else if (g==max) hue = 126 + 42*int(double(b-r)/double(delta));
            else if (b==max) hue = 210 + 42*int(double(r-g)/double(delta));
        }
        
        hsi[3*i] = hue;
        if(max==0) hsi[3*i+1] = 0;
        else hsi[3*i+1] = 255 - 3*int(255.0*double(min)/double(r+g+b));
        hsi[3*i+2]= (r+g+b)/3;
    }
}

void HSI2BIN(BYTE* hsi, BYTE* bin)
{   int i;
    int hue, saturation, intensity;
    for(i=0; i<PIXELS; i++)
    {   hue = hsi[3*i];
        saturation = hsi[3*i+1];
        intensity = hsi[3*i+2];
        
        bin[i] = BACKGROUND;
        //if( ((0<=hue&&hue<=25)||(325<=hue&&hue<=360))&&(saturation>=50)&&(intensity>25) ) bin[i] = 0; // stop signs
        if( ((0<=hue&&hue<=45)||(325<=hue&&hue<=360))&&(saturation>=50)&&(intensity>25) ) bin[i] = FOREGROUND; // stop signs
        else if( (35<=intensity&&intensity<=75)&&(180<=hue&&hue<=250) ) bin[i] = FOREGROUND; // parking
        else if ( (intensity>130)&&(saturation<=50) ) bin[i] = FOREGROUND; //white
    }
}

void classification(list<ROI>* _ROIs, list<Detection_Profile>* _profiles) {
    for(int i=0; i< MAX_CLASSIFICATIONS_PER_FRAME; i++) {
        int index;
        double confidence;
        
        list<Detection_Profile>::iterator it1;
        for (it1=_profiles->begin(); it1!=_profiles->end(); it1++) {
            if(it1->ticks_left<1) {
                ROI roi =  ROI(it1->x,it1->y,it1->xs,it1->ys);
                char *img_data = (char *)rgbMat.data;
                int len = SIZE;
                PyObject *pValue1 = PyMemoryView_FromMemory(img_data,len,PyBUF_READ);
                PyObject *pArgs = PyTuple_New(5);
                if (!pValue1) {
                    Py_XDECREF(pValue1);
                    Py_DECREF(pArgs);
                    Py_DECREF(pModule);
                    fprintf(stderr, "Cannot convert argument\n");
                    return;
                }
                PyTuple_SetItem(pArgs, 0, pValue1);
                PyTuple_SetItem(pArgs, 1, PyLong_FromLong(roi.x)); // set x
                PyTuple_SetItem(pArgs, 2, PyLong_FromLong(roi.y)); // set y
                PyTuple_SetItem(pArgs, 3, PyLong_FromLong(roi.xs)); // set xs
                PyTuple_SetItem(pArgs, 4, PyLong_FromLong(roi.ys)); // set ys
                
                PyObject *pValue2 = PyObject_CallObject(pFunc, pArgs);
                if (pValue2 != NULL) {
                    index = int(PyLong_AsLong(PyTuple_GetItem(pValue2,0)));
                    printf("label=%d\n", index);
                    confidence = PyFloat_AsDouble(PyTuple_GetItem(pValue2,1));
                    Py_XDECREF(pValue2);
                }
                else {
                    Py_XDECREF(pValue2);
                    PyErr_Print();
                    fprintf(stderr,"Call failed\n");
                    return;
                }
                
                Detection* observed = new Detection(index, confidence,roi.x,roi.y,roi.xs,roi.ys, roi.cluster);
                it1->add_detection(observed);
                continue;
            }
        }
        
        if(_ROIs->empty()) return;
        ROI *roi = &_ROIs->front();
        
        PyObject *pArgs = PyTuple_New(5);
        char *img_data = (char *)rgbMat.data;
        int len = SIZE;
        
        PyObject *pValue1 = PyMemoryView_FromMemory(img_data,len,PyBUF_READ);
        if (!pValue1) {
            Py_XDECREF(pValue1);
            Py_DECREF(pArgs);
            Py_DECREF(pModule);
            fprintf(stderr, "Cannot convert argument\n");
            return;
        }
        PyTuple_SetItem(pArgs, 0, pValue1);
        PyTuple_SetItem(pArgs, 1, PyLong_FromLong(roi->x)); // set x
        PyTuple_SetItem(pArgs, 2, PyLong_FromLong(roi->y)); // set y
        PyTuple_SetItem(pArgs, 3, PyLong_FromLong(roi->xs)); // set xs
        PyTuple_SetItem(pArgs, 4, PyLong_FromLong(roi->ys)); // set ys
        
        PyObject *pValue2 = PyObject_CallObject(pFunc, pArgs);
        if (pValue2 != NULL) {
            index = int(PyLong_AsLong(PyTuple_GetItem(pValue2,0)));
            printf("label=%d\n", index);
            confidence = PyFloat_AsDouble(PyTuple_GetItem(pValue2,1));
            Py_XDECREF(pValue2);
        }
        else {
            Py_XDECREF(pValue2);
            PyErr_Print();
            fprintf(stderr,"Call failed\n");
            return;
        }
        
        Detection* observed = new Detection(index, confidence,roi->x,roi->y,roi->xs,roi->ys, roi->cluster);
        Detection_Profile* dp = new Detection_Profile(observed);
        _profiles->push_back(*dp);
        
        _ROIs->pop_front();
    }
    return;
}

void show_detections(list<Detection_Profile>* _profiles) {
    list<Detection_Profile>::iterator it1;
    for (it1=_profiles->begin(); it1!=_profiles->end(); it1++) {
        int _class = (*it1).detections.back()->object_class;
        if(_class==0) continue;
        
        const static float X_M= 0.18;
        const static float X_C= -2.3;
        const static float Y_M= 0.06;
        const static float Y_C= 1.4;
        
        LCDArea(it1->x,it1->y,it1->x+it1->xs-1,it1->y+it1->ys-1,YELLOW,0);
        
        LCDSetPos(Y_M*it1->y+Y_C,X_M*it1->x+X_C);
        LCDPrintf("%s:%f",labels[(*it1).detections.back()->object_class].c_str(),(*it1).detections.back()->confidence);
    }
    return;
}

void conclude_frame(list<Detection_Profile>* _profiles) {
    list<Detection_Profile>::iterator it1;
    for (it1=_profiles->begin(); it1!=_profiles->end(); it1++) {
        it1->tick();
        
        printf("==================================\n");
        printf("object: %s\n", labels[(*it1).predicted_class].c_str());
        printf("reliability: %f\n", (*it1).reliability);
        printf("N: %d\n",it1->N_d );
        printf("tl: %d\n", it1->ticks_left);
        
        printf("table:\n");
        it1->printTable();
        printf("==================================\n");
        printf("\n");
        
        
    }
    return;
}
