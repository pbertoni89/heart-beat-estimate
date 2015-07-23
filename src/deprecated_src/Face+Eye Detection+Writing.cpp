// !!!!!!!!!!!
// ATTENZIONE: INCLUDERE ANCHE LA LIBRERIA opencv_objdetect220d.lib
// GLI XML UTILIZZATI SONO COPIE DI QUELLI LOCATI IN /opencv_220
// !!!!!!!!!!!

#include "stdafx.h"
#include <stdio.h>
#include <cv.h>
#include <highgui.h>
 
CvHaarClassifierCascade *cascadeFace, *cascadeEye;
CvMemStorage            *storage;
 
void detectEyes(IplImage *img, char* windowName, char* roiWindowName, CvVideoWriter* videoWriter);
void drawBox(IplImage *img, CvRect *box, int r, int g, int b);
 
int main(int argc, char** argv)
{ 

    CvCapture *capture;
	CvVideoWriter *videoWriter;
    IplImage  *frame;
    int       key = 0;

	char* windowName = "Eye Detection!";
	char* roiWindow = "Region of Interest";
   
	char *cascadeFacePath = "C:\\Users\\patrizio\\Documents\\Visual Studio 2010\\Projects\\BVP Thesis\\Other\\HAAR Cascades\\frontalface_alt.xml";
	char *cascadeEyePath = "C:\\Users\\patrizio\\Documents\\Visual Studio 2010\\Projects\\BVP Thesis\\Other\\HAAR Cascades\\eye.xml";

    char *videoCapturePath = "C:\\Users\\patrizio\\Documents\\Visual Studio 2010\\Projects\\BVP Thesis\\Other\\Videos\\4.avi";
	char *videoSavePath = "C:\\Users\\patrizio\\Documents\\Visual Studio 2010\\Projects\\BVP Thesis\\Other\\Videos\\Outputs\\4eye.avi";

    cascadeFace = (CvHaarClassifierCascade*) cvLoad(cascadeFacePath, 0, 0, 0 );
    assert(cascadeFace);
	cascadeEye = (CvHaarClassifierCascade*) cvLoad(cascadeEyePath, 0, 0, 0 );
    assert(cascadeEye);

	capture = cvCaptureFromAVI(videoCapturePath);
	assert(capture);

    //setup memory buffer; needed by the face detector
    storage = cvCreateMemStorage(0);
	assert(storage); 

    cvNamedWindow(windowName, 1);
	cvNamedWindow(roiWindow,CV_WINDOW_AUTOSIZE);
 
	CvSize size = cvSize((int)cvGetCaptureProperty( capture, CV_CAP_PROP_FRAME_WIDTH),
			(int)cvGetCaptureProperty( capture, CV_CAP_PROP_FRAME_HEIGHT));
	videoWriter = cvCreateVideoWriter(videoSavePath, -1 , 10 , size, 1);

    while(key!='q') {

        frame = cvQueryFrame(capture);

        if(!frame) 
			break;
 
        //'fix' frame. perchè ribaltare? no.
        //cvFlip( frame, frame, -1 );
        frame->origin = 0; // le coordinate partono da alto sx; se era 1, era basso dx
 
        detectEyes(frame, windowName, roiWindow, videoWriter); //frame, lena
		//system("PAUSE");
        key = cvWaitKey(1);
    }
 
    cvReleaseCapture(&capture);
	cvReleaseVideoWriter(&videoWriter);
    cvDestroyWindow(windowName);
	cvDestroyWindow(roiWindow);
    cvReleaseHaarClassifierCascade(&cascadeFace);
	cvReleaseHaarClassifierCascade(&cascadeEye);
    cvReleaseMemStorage( &storage );
 
    return 0;
}
 
void detectEyes(IplImage *frame, char* windowName, char* roiWindowName, CvVideoWriter* videoWriter) {

    int i,j;
    CvSeq *faces, *eyes;
	CvRect *face, *eye;
	
	faces = cvHaarDetectObjects(frame, cascadeFace, storage, 1.1, 3, 0, cvSize( 40, 40 ) );
 
    // for each face found, draw a red box 
    for(i=0; i<(faces?faces->total:0); i++) {

        face = (CvRect*) cvGetSeqElem( faces, i );
        drawBox(frame, face, 255, 0, 0);

		// Set the Region of Interest: estimate the eyes' position
		cvSetImageROI(
			frame,                   
			cvRect(
				face->x + (int)(face->width/9.5),            // x = start from leftmost
				face->y + (int)(face->height/5.5), // y = a few pixels from the top
				(int)(face->width/1.2),        // width = same width with the face
				(int)(face->height/3.0)    // height = 1/3 of face height
			)
		); 

		IplImage *roi =  cvCreateImage(cvGetSize(frame),frame->depth,frame->nChannels);
		cvCopy(frame,roi, NULL);
//cvShowImage(roiWindowName,roi);

		eyes = cvHaarDetectObjects(frame, cascadeEye, storage, 1.15, 3, CV_HAAR_DO_CANNY_PRUNING, cvSize(25, 15) );
		// for each eye found, draw a blue box 
		for(j=0; j<(eyes?eyes->total:0); j++) {
			eye = (CvRect*) cvGetSeqElem( eyes, j );
			drawBox(frame, eye, 0, 0, 255);
		}

	}
	cvResetImageROI(frame);
//cvShowImage(windowName, frame); 
	cvWriteFrame(videoWriter, frame); // ovviamente non deve stare qua!!!
}
 
void drawBox(IplImage *img, CvRect *box, int r, int g, int b) {
	cvRectangle( img,
                     cvPoint( box->x, box->y ),
                     cvPoint( box->x + box->width, box->y + box->height ),
                     CV_RGB(r,g,b), 1, 8, 0 );
}
