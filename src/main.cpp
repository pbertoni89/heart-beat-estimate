#pragma once
#include "common_cfg.h"
#include "ProcessBlock.h"
#include <highgui.h>

using namespace std;
using namespace cv;

//------------------------------------------------------------------------------------------------------------------------------
// Precompiler Flags
//------------------------------------------------------------------------------------------------------------------------------
#define WINTIMESTAMP   // prints nice timestamps to track fps performances for face detection
					   // TAKE CARE! This feature surely works at least on Windows 7 x86 under Visual Studio 2010 compiler.
#ifdef WINTIMESTAMP
	#include <time.h>
	#include <sys\types.h>
	#include <sys\timeb.h>
#endif

#define DONOTCLOSEENGINE // keeps Matlab Engine Opened to perform faster subsequent startups
//#define SRC_FACE       // sets face patterns as source for BVP extracting. 
					     // #undefine to get BVP from finger (must press it gently on the camera)
#define RES_1280_720	 // Resolution imposed to device. 1280_720 may have unattended consequences if not supported! Default is 640x480
#define WIN_640_480		 // Impose window dimensions to 640x480. This will clearly wont have effects on processing.
#define POSX 690		 // Default coordinates on my monitor to show window.
#define POSY 16

//------------------------------------------------------------------------------------------------------------------------------
// Global Variables and Constants
//------------------------------------------------------------------------------------------------------------------------------
#ifdef RES_1280_720
	#define FRAMEH 720
	#define FRAMEW 1280
#else
	#define FRAMEH 480
	#define FRAMEW 640
#endif
#define EXITKEY 27     // hotkey to stop frame querying (27 = 'ESC'). Valid in both Camera and AVI modes.
#ifdef SRC_FACE
	#define HAARCACHE 4    // maximum limit of subsequent unsuccessful face recognitions in frame stream. (unuseful if HAARSTRICT == 1, but nvm)
	#define HAARSTRICT 3   // 1 must retrieve at least one face at first frame; for the rest of the video, haar cache is unlimited. 
					       // 2 forces face recognition to return R.o.I. at every frame, admitting HAARCACHE subsequent exceptions. (DEFAULT)
					       // 3 caches as in #2, but checks detection consistency by confronting difference from new coords found and old coords, 
						        //with a fixed treshold. If below treshold, it means that face has not moved and we can consider its (width,height)
								//stationary, and impose the old values. We hope to avoid fake movement peaks in this way.
	#if HAARSTRICT == 3
		#define HAARTRESH 8 //random experimental value!
	#endif
	#define NFACE 3         // similar faces to be recognized for neighborhood. More is better but slower.
#endif

#define FRAMEBLOCK (500 + 1 + FRAMEDROP)
					// Rough number of frames a block should be consisted of. Too, maximum number of frames considered in every processing cycle.
					// Is source is CAMERA, blocks will always be of FRAMEBLOCK cardinality. Else, in AVI source mode, the last block processed may be
					// of a minor cardinality. In order to get the final cardinality of a Process Block, to FRAMEBLOCK you should subtract:
					// -1 frame  (because difference series of N samples intrinsecally consist of N-1 samples)
					// -FRAMEDROP (see FRAMEDROP macro into ProcessBlock.h)

#define OVERLAP 250 // indicates how much frames are considered old and get popped out from queue at every process cycle. 
					// NOT tested what happens if precompiled with OVERLAP > NETFRAMEBLOCK, you should never do this!
//------------------------------------------------------------------------------------------------------------------------------
// Strings for Video Output
//------------------------------------------------------------------------------------------------------------------------------
char *face_cascade_path = "C:\\Users\\patrizio\\Documents\\Visual Studio 2010\\Projects\\HRSolution\\External\\frontalface_alt.xml";
char *video_capture_path = "C:\\Users\\patrizio\\Desktop\\Thesis_videos\\pat.avi"; //pat //BVPme
char* hello_string = "Welcome to Heart Rate Estimate!\nYou can exit program by pressing ESC into video window during capturing.\n";
char* hello_init = "\n\nInitializing...\n";
char *matlab_setup = "\tSetting up MATLAB Engine connection...\n";
char *matlab_conn_ok = "\tConnection with Matlab Engine opened.\n";
char *matlab_conn_fail = "Sorry, can't start MATLAB engine. Aborting Matlab connection.\n";
char *matlab_rdy_close = "\n\tData fully plotted. Ready to close MATLAB session.\n\n";
char* error_query_frame = "\nError retrieving some frame. Check source.\n";
char* error_malloc = "\nCannot allocate sufficient memory for computations.\n";
char* error_cap = "\nThere were problems opening video capture.\n";
char* error_stor = "\nThere were problems opening storage structure. Check RAM state.\n";
char* error_cascade = "\nThere were problems opening cascade XML file. Check filesystem.\n";
char *ok_avi_cap = "\tVideo capture successfully opened from AVI source.\n";
char *ok_cam_cap = "\tVideo capture successfully opened from CAM source.\n";
char* error_haar_detect = "\n\nNo face recognized at first frame, or\n\tface tracking lost for too many frames, or\n\tface exited from camera range of view.\nCannot cache anymore, neither skip this point. Consider:\n\trecording a better quality video\n\tchange precision parameters\n\tstand steady in range of view!\n"; 
char* buff_start = "\n\tBuffering started...\t";
char* buff_cont = "\n\tContinue buffering...\t";
char* frame_tot = "\n\tTotal number of frames is ";
char* frame_block = "\n\tFrame Block size is ";
char* frame_overlap = "\tFrame Block overlap is ";
char* fps_text = "\n\tAverage FPS while detecting face is ";
char* proc_start = "\n\n\tProcessing started...  ";
char* proc_remn = "\n\n\tProcessing started for remaining frames...\n";
char* exit_string = "\n\n\nHRestimate exited by user request. Goodbye!\n";
char* error_string = "\nExiting with -1...\n";
char* goodbye_string = "\n\n\nHRestimate exited with success. Goodbye!\n";
char* cap_win = "Video Capture";
//------------------------------------------------------------------------------------------------------------------------------
// Structure of main Services
//------------------------------------------------------------------------------------------------------------------------------
typedef struct {
	
	CvCapture *capture;					// Pointer to OpenCV Video Capture.
	#ifdef SRC_FACE
		CvMemStorage *storage;				// Buffer for Face Recognition related data.
		CvHaarClassifierCascade *cascade;	// Cascade HAAR Classifier.
	#endif
	#ifdef MATLAB
		Engine *engine;			        // Pointer to Matlab Engine.	
	#endif

} Services, *pServices;
//------------------------------------------------------------------------------------------------------------------------------
// Function Declaration
//------------------------------------------------------------------------------------------------------------------------------
	void init_prog(pServices servs);
	bool routine_cycle(pServices servs, vector<Mat*>* faces_buf, int *localIndex, double *lastfps);
	void handleProcessCall(vector<Mat*>* faces_buf, double avgfps, double *lastfps
				#ifdef MATLAB 
				 ,Engine *engine
				#endif
				);

#ifndef CAMERA
	void last_cycle(vector<Mat*>* faces_buf, int localIndex, double lastfps
					#ifdef MATLAB 
					 ,Engine *engine  //check if we can pass only struct fields
					#endif
					);
#endif
	void clean(pServices servs);
	void welcome(void);
	void salute(short exitval);
	CvCapture* openCapture(void);
#ifdef SRC_FACE
	CvRect detectFace(IplImage *pframe, CvHaarClassifierCascade *cascadeFace, CvMemStorage *storage);
#endif
	bool display(IplImage* pframe, CvRect *rect
				 #ifdef WINTIMESTAMP 
				  ,double instfps
				 #endif
				 );
#ifdef WINTIMESTAMP
	double getfps(timeb t1, timeb t2);
#endif
	void presentOutput(double v[]);
	void writeProgress(int local, int step10);
#ifdef MATLAB
	Engine* openMatlab(void);
	void closeMatlab(Engine *e, bool last); //check if we can pass only struct fields
#endif
	void handleError(char* text);
//------------------------------------------------------------------------------------------------------------------------------
// Main Program
//------------------------------------------------------------------------------------------------------------------------------
int main(int argc, char* argv[]) {

	welcome();
	
	short exitval = NORMALSTATE;	 // Unique flag for the state of program. See macros for detailed explanation.
	int localIndex = 0;				 // Index used for walking on the buffer. Shared variable between routine_cycle() and last_cycle() (if compiled)
	double lastfps = FPS;			 /* Shared variable between routine_cycle() and last_cycle() (if compiled)
										Last avgfps computed, kept for last_cycle() (we assume fps will be stationary on the machine)
										Comes from a too simple avgfps compute algorithm (we sum instantfps weighted (divided by FRAMEBLOCK))
										If WINTIMESTAMP is undefined, lastfps will simply remain equal to FPS. */
	vector<Mat*> buffer(FRAMEBLOCK); // Vector of pointers to frames buffered.

	pServices main_serv = new Services; // Data structure containing the core variables of HRestimate.
	
	init_prog(main_serv);
system("PAUSE");
	if(routine_cycle(main_serv, &buffer, &localIndex, &lastfps)) {
							#ifndef CAMERA 
							last_cycle(&buffer, localIndex, lastfps
										#ifdef MATLAB 
										, main_serv->engine
										#endif
										);
							#endif
	} else
		exitval = EXITSTATE;

	salute(exitval);
	clean(main_serv);
   
	return exitval;
}
//------------------------------------------------------------------------------------------------------------------------------
// Procedures
//------------------------------------------------------------------------------------------------------------------------------
void init_prog(pServices servs) {

	servs->capture = NULL;  
	servs->capture = openCapture();

	#ifdef SRC_FACE
		servs->storage = NULL; 
		if(!(servs->storage = cvCreateMemStorage()))
			handleError(error_stor);

		servs->cascade = NULL;
		if(!(servs->cascade = (CvHaarClassifierCascade*) cvLoad(face_cascade_path, servs->storage, 0, 0 )))
			handleError(error_cascade);
	#endif

	#ifdef MATLAB
		servs->engine = NULL;
		servs->engine = openMatlab();
	#endif

	cvNamedWindow(cap_win, CV_WINDOW_NORMAL);   //cvNamedWindow(cap_win, CV_WINDOW_AUTOSIZE); will follow frame dimensions every time
	#ifdef WIN_640_480
		cvResizeWindow(cap_win, 640, 480);
	#endif
	cvMoveWindow(cap_win, POSX, POSY);
}
//------------------------------------------------------------------------------------------------------------------------------
bool routine_cycle(pServices servs, vector<Mat*>* faces_buf, int *localIndex, double *lastfps) {

	bool first = true;  // indicates whether this is or not the first capture cycle. Flag needed for certain operations to reduce number of checks.
	bool normal = true;             // propagates toward extern the will of continue video reading.
	int fresh = FRAMEBLOCK-OVERLAP; // indexes for walking on frame buffer.
	int step10 = FRAMEBLOCK/10;  // number of frames to reach a 10% percentage of total. Used for display process purposes.

	double avgfps;               // average frames per second.
	#ifdef WINTIMESTAMP
		double instfps = 0;
		double rstfps = (double)fresh/FRAMEBLOCK; // corrects weighted value from which average approximation should continue
		avgfps = 0;
		timeb time1, time2;      //timers for fps while capturing.
		const int instdrops = 4; // discards instfps produced by first #instdrops frames of every capture cycle: 
								 // they have awful massive values, of unknown source. this is a very rough bugfix, please analyze it further!
	#else
		avgfps = FPS;       //since real fps cannot be computed.
	#endif

	Mat face, *pface;      // Mat representing the frame restricted to face area.
	IplImage *ipl = NULL;  // Pointer to IplImage representing the frame queried at every time.
	CvRect roi;			   // Rectangle pointing to the area covered by recognized face.
	
	#ifndef CAMERA
		int frameTot = (int) cvGetCaptureProperty(servs->capture, CV_CAP_PROP_FRAME_COUNT);  // total counter of frames to compute.
		int frameCount = 0;												   	                 // total counter of frames computed.
		cout<<frame_tot<<frameTot<<endl;	
	#endif
		cout<<buff_start;
		
	while(normal
			#ifndef CAMERA
			&& frameCount<frameTot
			#endif	
			) { 

		#ifdef WINTIMESTAMP
			ftime(&time1);
		#endif

					if(!(ipl = cvQueryFrame(servs->capture)))
						handleError(error_query_frame);

					writeProgress(*localIndex, step10);
					#ifdef SRC_FACE
						roi = detectFace(ipl, servs->cascade, servs->storage);
						face = Mat(Mat(ipl).clone()(roi)); 
								/* constructs a Mat header "face" from another Mat "Mat(ipl).clone", restricted to "roi".
								   "Mat(ipl).clone" is a deep copy of another Mat obtained from IplImage "ipl".
								   deep copy is needed for a) leave frame untouched for displaying; b) get non null dimensions. */
					#else
						face = Mat(Mat(ipl).clone());
					#endif
					if(!(pface = new Mat))
						handleError(error_malloc);
		
					*pface = face;
					faces_buf->at(*localIndex) = pface;
					(*localIndex)++;
	
		#ifdef WINTIMESTAMP
			ftime(&time2); 
			instfps = getfps(time1, time2); 	
			if(first) {
				if(*localIndex>instdrops) 
					avgfps += (instfps/(FRAMEBLOCK-instdrops));
			} else
				if(*localIndex-fresh>=instdrops)
					avgfps += (instfps/(FRAMEBLOCK-instdrops));
			normal = display(ipl, &roi, instfps);
		#else
			normal = display(ipl, &roi);
		#endif-

		if(*localIndex==FRAMEBLOCK) {
			handleProcessCall(faces_buf, avgfps, lastfps
			#ifdef MATLAB 
				 , servs->engine
			#endif
			);
			first = false;
			*localIndex = fresh;
			avgfps *= rstfps;
		 }
		#ifndef CAMERA
			frameCount++;
		#endif
	}
	return normal;
}
//------------------------------------------------------------------------------------------------------------------------------
void handleProcessCall( vector<Mat*>* faces_buf, double avgfps, double *lastfps
				#ifdef MATLAB 
				 ,Engine *engine
				#endif
				) {

	#ifdef WINTIMESTAMP
		cout<<fps_text<<avgfps<<endl;
		*lastfps = avgfps;
	#endif

	cout<<proc_start;
	queue<Mat> faces_q; 

	for(int j=0; j<FRAMEBLOCK; j++)
		faces_q.push(*faces_buf->at(j));
	for(int l=0; l<OVERLAP; l++)
		delete faces_buf->at(l);
	for(int k=OVERLAP; k<FRAMEBLOCK; k++)
		faces_buf->at(k-OVERLAP) = faces_buf->at(k);
			
	ProcessBlock proc(faces_q, avgfps, false
						#ifdef MATLAB
						, engine
						#endif
						);
	proc.process();
	presentOutput(proc.getHR());
	#ifdef MATLAB
		closeMatlab(engine, false);   
	#endif
	#ifdef WINTIMESTAMP
		avgfps = 0;
	#endif
	cout<<buff_cont; //proc.~ProcessBlock(); CHIAMATO DA SOLO!!
}
//------------------------------------------------------------------------------------------------------------------------------
#ifndef CAMERA
void last_cycle(vector<Mat*>* faces_buf, int localIndex, double lastfps
				#ifdef MATLAB 
				 ,Engine *engine
				#endif
				) {

	#if VERBOSE >= 1
		cout<<proc_remn;
	#endif	
	queue<Mat> last_q;
					
	for(int i=0; i<localIndex; i++) {
		last_q.push( *faces_buf->at(i) );
		faces_buf->at(i)->release();
	}
	// WRN: actually it keeps last value computed for realfps. We assume this is stationary once program started so all right
	ProcessBlock proc(last_q, lastfps, false
								#ifdef MATLAB
								, engine
								#endif
								);
	proc.process();
	presentOutput(proc.getHR());

	while(last_q.size())
		last_q.pop();
}
#endif
//------------------------------------------------------------------------------------------------------------------------------
void clean(pServices servs) {

	#ifdef MATLAB
		#ifndef DONOTCLOSEENGINE
			closeMatlab(servs->engine, true);
		#else
			engEvalString(servs->engine, "clear; close all;");
		#endif
	#endif
	cvReleaseCapture(&(servs->capture));
	cvDestroyAllWindows();
	#ifdef SRC_FACE
		cvReleaseHaarClassifierCascade(&(servs->cascade)); 
		cvReleaseMemStorage(&(servs->storage));
	#endif

	delete servs;
}
//------------------------------------------------------------------------------------------------------------------------------
//		Face Recognition Functions
//------------------------------------------------------------------------------------------------------------------------------
/* Loads an AVI video file or a Camera connection, and checks for success. */
CvCapture* openCapture(void) {

	#ifdef CAMERA
		//CvCapture* cap = cvCaptureFromCAM(CV_CAP_ANY); // if you want to connect a laptop integrated webcam, this will probably work
		CvCapture* cap = cvCaptureFromCAM(1); // if you want to connect an usb camera to a laptop which already has a webcam, this will probably work
	#else 
		CvCapture* cap = cvCaptureFromAVI(video_capture_path);
	#endif
	if (!cap)
		handleError(error_cap);
	else {
		#if VERBOSE >= 1
			#ifdef CAMERA
				cout<<ok_cam_cap;
			#else
				cout<<ok_avi_cap;
			#endif
		#endif
	}
	cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, FRAMEW);
	cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, FRAMEH);

	short w = (short) cvGetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH), h = (short) cvGetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT);
	cout<<"\tCapture Resolution is "<<w<<" x "<<h<<endl<<endl;
	
	return cap;
}
//------------------------------------------------------------------------------------------------------------------------------
#ifdef SRC_FACE
/* Loads video and looks for a rectangular Region of Interest representing a human face, using a Haar Cascade Classifier.
   Exites with special value if problems during opening classifier xml file. */
CvRect detectFace(IplImage *pframe, CvHaarClassifierCascade *cascadeFace, CvMemStorage *storage) {
	
	static CvRect *rect = NULL;
	#if HAARSTRICT == 3
		static CvRect *old = NULL;
		static double normold; // norm of the vector (old->x, old->y);
	#endif
	int precision = 100; //40 in web examples. less is faster
	const static CvSize minsize = cvSize( precision, precision );
	static int valid = HAARCACHE;
	Mat gray;

	cvtColor(Mat(pframe), gray, CV_BGR2GRAY);
	equalizeHist(gray, gray);

	IplImage iplgray = gray;
		
	CvSeq *rects = cvHaarDetectObjects(&iplgray, cascadeFace, storage, 1.2, NFACE, CV_HAAR_DO_CANNY_PRUNING, minsize);

	if(!rects->total)
		if(!rect||!valid)
			handleError(error_haar_detect);
		else {
			#if HAARSTRICT >= 2	
				valid -= 1;
			#endif
			}
	else {
		#if HAARSTRICT >= 2
			valid = HAARCACHE;
		#endif
		rect = (CvRect*) cvGetSeqElem(rects, 0); //takes always first face (CvSeq should be loaded randomly, as far as i know)
		rect->x      += (int) (0.1*rect->width);  
		rect->y      -= (int) (0.1*rect->height);
		rect->height += (int) (0.16*rect->height); 
		rect->width  -= (int) (0.16*rect->width); // stretching roi with magic numbers
		#if HAARSTRICT == 3
			if(!old)
				old = rect;
			normold = cv::norm(Point(old->x,old->y));
			double normrect = cv::norm(Point(rect->x,rect->y));
			if(fabs(normrect-normold)<HAARTRESH) {
				if(old->x<0||old->y<0)
					handleError(error_haar_detect);
				return *old; }
			else
				old = rect;
		#endif
	}
	if(rect)
		if(rect->x<0||rect->y<0)
				handleError(error_haar_detect);
	return *rect;
}
#endif
//------------------------------------------------------------------------------------------------------------------------------
/* Displays in output video window one frame at once, plus a red box around the detected face in it.
   Returns false if ESC key is pressed during cvWaitKey() waiting for input. */
bool display(IplImage* pframe, CvRect *rect
				 #ifdef WINTIMESTAMP
				 , double instfps
				 #endif
				 ) {

	#ifdef SRC_FACE
		int thick = 1;
		cvRectangle( pframe,
					 cvPoint( rect->x-thick, rect->y-thick ), //avoid displaying red line
					 cvPoint( rect->x + rect->width, rect->y + rect->height ),
					 CV_RGB(255,0,0), thick, 8, 0 );
	#endif
	 
	#ifdef WINTIMESTAMP
		char numstr[21]; // enough to hold all numbers up to 64-bits
		sprintf_s(numstr, "%.2f", instfps);
		string result = string("fps: ") + numstr;
		CvFont font;
		cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0, 1, CV_AA);
		cvPutText(pframe, result.data(), cvPoint(2,30), &font, CV_RGB(255,255,255));
	#endif

	cvShowImage(cap_win, pframe); 

	if(cvWaitKey(5)==EXITKEY)  //WRN WRN WRN WRN WRN WRN WRN WRN WRN WRN: clock element for the whole program
		return false;
	else
		return true;
}
//------------------------------------------------------------------------------------------------------------------------------
#ifdef WINTIMESTAMP
/* Prints out frames computed per second during capture */
double getfps(timeb t1, timeb t2) {
	double secs = (double) t2.time - t1.time;
	double mills = t2.millitm - t1.millitm;
	secs += mills/1000;
	return 1/secs;
}
#endif
//------------------------------------------------------------------------------------------------------------------------------
//		Matlab Functions
//------------------------------------------------------------------------------------------------------------------------------
#ifdef MATLAB
// Handle the opening and subsequent connection with Matlab Engine.
// Can set Program Exitvalue = -1, because Engine is mandatory for the correctness of whole program.
Engine* openMatlab(void){
	
	#if VERBOSE >= 1
		cout<<matlab_setup;
	#endif
	Engine *e;
	if (!(e = engOpen(NULL))) 
		handleError(matlab_conn_fail);
	else {
		engEvalString(e, "clear all; close all; clc;");
		#if VERBOSE >= 1
			cout<<matlab_conn_ok;
		#endif
	}
	return e;
}
//------------------------------------------------------------------------------------------------------------------------------
// Closes all Matlab figures and, if exit, shuts down Matlab Engine too.
void closeMatlab(Engine *e, bool last) {
	if(e)
		if(last) {
			cout<<matlab_rdy_close;
			system("PAUSE");
			engEvalString(e, "close all; exit;");
		 }
		else {
			system("PAUSE");
			engEvalString(e, "close all;");
		 }
}
#endif
//------------------------------------------------------------------------------------------------------------------------------
//		User Interaction Functions
//------------------------------------------------------------------------------------------------------------------------------
void handleError(char* text) {
	cout<<text; 
	cout<<error_string;
	exit(ERRORSTATE);
}
//------------------------------------------------------------------------------------------------------------------------------
void welcome(void) {
	cout<<hello_string;
	cout<<hello_init;
	#if VERBOSE >= 2
		cout<<frame_block<<FRAMEBLOCK<<endl;
		cout<<frame_overlap<<OVERLAP<<endl;
	#endif 
}
//------------------------------------------------------------------------------------------------------------------------------
void salute(short exitval) {
	if(exitval==EXITSTATE)
		cout<<exit_string;
	else
		cout<<goodbye_string;
}
//------------------------------------------------------------------------------------------------------------------------------
void presentOutput(double v[]) { 
	cout<<"\n\n\tThe Heart Rate is (Channel 1): "<< v[0] <<" bpm";
	cout<<  "\n\tThe Heart Rate is (Channel 2): "<< v[1] <<" bpm";
	cout<<  "\n\tThe Heart Rate is (Channel 3): "<< v[2] <<" bpm\n\n";
}
//------------------------------------------------------------------------------------------------------------------------------
void writeProgress(int local, int step10) {
	int num = 0;
	if(!(local%step10) )
		num = local/step10;
	if(num)
		cout<<" "<<num<<"0%";
}