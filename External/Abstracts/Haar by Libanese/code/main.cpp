/**
 *  Lab4: Face-Detection
 *  @author Fabrizio Pedersoli 72816
 *  @author Mohmad Farran 73606
 *  @file main.cpp
 */

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
using namespace std;

int main (int argc, char *argv[])
{

  if (argc != 4)
    {
      fprintf(stderr, "./main src dest min\n");
      exit(-2);
    }

  char *file = (char*)argv[1];
  char *output = (char*)argv[2];
  int size = atoi(argv[3]);
  char win[] = "Haar window";

  IplImage *pInpImg;
  CvHaarClassifierCascade *pCascade = 0;
  CvMemStorage *pStorage = 0;
  CvSeq *pFaceRectSeq;
  int i;

  pInpImg = cvLoadImage(file, 1);
  //cvEqualizeHist(pInpImg, pInpImg);

  pStorage = cvCreateMemStorage(0);
  pCascade = (CvHaarClassifierCascade*)cvLoad(
  "/Users/fab/mylibs/share/opencv/haarcascades/haarcascade_frontalface_alt2.xml");

  if (!pInpImg || !pStorage || !pCascade)
    {
      fprintf(stderr, "Unalbe to laod something :)\n");
      exit(-1);
    }

  int flag = CV_HAAR_FIND_BIGGEST_OBJECT;
  pFaceRectSeq = cvHaarDetectObjects (pInpImg, pCascade, pStorage,
				      1.1,
				      3,
				      flag,
				      cvSize(size,size));
 
   
  for (i=0; i<(pFaceRectSeq? pFaceRectSeq->total:0); i++)
    {
      CvRect *r = (CvRect*)cvGetSeqElem(pFaceRectSeq, i);
      CvPoint pt1 = {r->x, r->y};
      CvPoint pt2 = {r->x + r->width, r->y + r->height};
      cvRectangle(pInpImg, pt1, pt2, CV_RGB(0,255,0), 3, 3, 0);
    }

  /*
  cvNamedWindow(win, CV_WINDOW_AUTOSIZE);
  cvShowImage(win, pInpImg);
  cvWaitKey(0);
  cvDestroyWindow(win);*/
  

  cvSaveImage(output, pInpImg);

  cvReleaseImage(&pInpImg);
  cvReleaseHaarClassifierCascade(&pCascade);
  cvReleaseMemStorage(&pStorage);

  return 0;
}




