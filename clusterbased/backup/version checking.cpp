#include <Python.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
int main(int argc, char** argv )
{
    //PyRun_SimpleString("import sys");
    char *app_name = (char *)"SIFT-like Keypoint Cluster-based Traffic Sign Recognition and Detection";
    Py_SetProgramName((wchar_t*)app_name);
    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("print (sys.version)");
    
    cout << "OpenCV version : " << CV_VERSION << endl;
    cout << "Major version : " << CV_MAJOR_VERSION << endl;
    cout << "Minor version : " << CV_MINOR_VERSION << endl;
    cout << "Subminor version : " << CV_SUBMINOR_VERSION << endl;
    
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    Mat image;
    image = imread( argv[1], 1 );
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);
    waitKey(0);
    return 0;
}
