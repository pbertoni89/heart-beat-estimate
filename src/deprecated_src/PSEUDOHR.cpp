
int main(int argc, char* argv[]) {

	video = open file(path, frameTot); //with assert(video)
	cascadeFace = (CvHaarClassifierCascade*) cvLoad("path", 0, 0, 0 );
	assert(cascadeFace);

	storage = cvCreateMemStorage(0);
	assert(storage); 
	cvNamedWindow("HRestimate", 1);
	queue <Mat> faces_buf;
	IplImage *frame; //Mat frame;
	Rect face;
	int i;

	while(i<frameTot) {

		frame = query(video); if(!frame) break;
		frame->origin = 0;
		face = detectFace(frame);
		faces_buf.push(frame(roi)); //queue gets entries by value, NOT by reference =)
		drawRedBox(frame, roi); //check that this WONT modify frames in buffer
		imshow(win, frame); //after displaying frame, roi are unuseful

		if(faces_buf.size()>FRAMEBLOCK) {
			{/* stuff on current block */}
			for(i=0; i<OVERLAP; i++)
				faces_buf.pop();
		}
		i++;
	}

	{/* do stuff with block dimension = frameTot%FRAMEBLOCK  (lastblock fraction part)*/}
			for(int j=0; frameTot%FRAMEBLOCK; j++)
				faces_buf.pop();
				

	cout<<"end main"<<endl;
	system("PAUSE");
	cvReleaseCapture(&capture);
	cvDestroyWindow(windowName);
	cvReleaseHaarClassifierCascade(&cascadeFace);
	cvReleaseMemStorage( &storage );

	return EXIT_SUCCESS;
}


void detectFace(IplImage *frame) {

    CvSeq *rects = cvHaarDetectObjects(frame, cascadeFace, storage, 1.1, 3, 0, cvSize( 40, 40 ) );
	CvRect *rect;

	//for(i=0; i<(rects?rects->total:0); i++) {
	for(int i=0; (i<(rects?rects->total:0))&&(i<=NEIGHBORS); i++) {
		rect = (CvRect*) cvGetSeqElem(rects, i);
		cvRectangle( img,
				 cvPoint( box->x, box->y ),
				 cvPoint( box->x + box->width, box->y + box->height ),
				 CV_RGB(255,0,0), 1, 8, 0 ); // pure red
	}		
}