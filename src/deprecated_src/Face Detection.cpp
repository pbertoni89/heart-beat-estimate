// !!!!!!!!!!!
// ATTENZIONE: INCLUDERE ANCHE LA LIBRERIA opencv_objdetect220d.lib
// GLI XML UTILIZZATI SONO COPIE DI QUELLI LOCATI IN /opencv_220
// !!!!!!!!!!!

#include "stdafx.h"
#include <stdio.h>
#include <cv.h>
#include <highgui.h>
 
CvHaarClassifierCascade *cascadeFace;
CvMemStorage            *storage;
 
void detectFace(IplImage *img, char* windowName);
void drawBox(IplImage *img, CvRect *box, int r, int g, int b);
 
int main(int argc, char** argv)
{ 
    CvCapture *capture;
    IplImage  *frame;
    int       key = 0;

	char* windowName = "Eye Detection!";
	char *cascadeFacePath = "C:\\Users\\patrizio\\Documents\\Visual Studio 2010\\Projects\\BVP Thesis\\Other\\HAAR Cascades\\frontalface_alt.xml";

    char *videoCapturePath = "C:\\Users\\patrizio\\Documents\\Visual Studio 2010\\Projects\\BVP Thesis\\Other\\Videos\\4.avi";

    cascadeFace = (CvHaarClassifierCascade*) cvLoad(cascadeFacePath, 0, 0, 0 );
    assert(cascadeFace);
	
	capture = cvCaptureFromAVI(videoCapturePath);
	assert(capture);

    //setup memory buffer; needed by the face detector. in EyeDetection program, it is used for eyes too
    storage = cvCreateMemStorage(0);
	assert(storage); 

    cvNamedWindow(windowName, 1);
 
    while(key!='q') {

        frame = cvQueryFrame(capture);

        if(!frame) 
			break;
 
        //cvFlip( frame, frame, -1 ); //'fix' frame. perchè ribaltare? no.
        frame->origin = 0; // le coordinate partono da alto sx; se era 1, era basso dx
 
        detectFace(frame, windowName);

        key = cvWaitKey(1);
    }
 
    cvReleaseCapture(&capture);
    cvDestroyWindow(windowName);
    cvReleaseHaarClassifierCascade(&cascadeFace);
    cvReleaseMemStorage( &storage );
 
    return 0;
}
 
void detectFace(IplImage *frame, char* windowName) {

    int i,j;
    CvSeq *faces;
	CvRect *face;
	
	faces = cvHaarDetectObjects(frame, cascadeFace, storage, 1.1, 3, 0, cvSize( 40, 40 ) );
 
    for(i=0; i<(faces?faces->total:0); i++) {

        face = (CvRect*) cvGetSeqElem( faces, i );
        drawBox(frame, face, 255, 0, 0);

	}
	cvShowImage(windowName, frame); 
}
 
void drawBox(IplImage *img, CvRect *box, int r, int g, int b) {
	cvRectangle( img,
                     cvPoint( box->x, box->y ),
                     cvPoint( box->x + box->width, box->y + box->height ),
                     CV_RGB(r,g,b), 1, 8, 0 );
}