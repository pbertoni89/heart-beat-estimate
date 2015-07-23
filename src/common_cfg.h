#include <iostream>
#include <cv.h>

#include <queue>       // used for frame buffering 

#define MATLAB		   // allows to ask Matlab software do a lot of amazing stuff, such as plotting graphs.
#ifdef MATLAB
	#include <engine.h>
	#pragma comment( lib, "c:\\users\\patrizio\\Desktop\\Matlab_Portable\\R2010b\\extern\\lib\\win32\\microsoft\\libmex.lib" )
	#pragma comment( lib, "c:\\users\\patrizio\\Desktop\\Matlab_Portable\\R2010b\\extern\\lib\\win32\\microsoft\\libeng.lib" )
	#pragma comment( lib, "c:\\users\\patrizio\\Desktop\\Matlab_Portable\\R2010b\\extern\\lib\\win32\\microsoft\\libmx.lib" )
#endif
#define VERBOSE 2	    // normal level is 1. Minimum level is 0. Deep level is 2
#define HUNGRY 1        // tries to clean memory as soon as possible, every time. Minimum 0

#define FPS 15          // frames per second at which the video should be played.

#define CAMERA          // sets up progam to dialog with camera instead of opening an AVI file.
#ifdef CAMERA
	#define FRAMEDROP 10 // Frames at starting of every cycle that are being dropped to avoid harmful discontinuities: 
#endif						// this happens only in CAMERA mode, since AVI video is intrinsecally continuous.
							

#define NORMALSTATE 0   // program is working correctly and will terminate with success.
#define EXITSTATE 1     // user called program closure by pressing 'ESC'. will terminate safely.
#define ERRORSTATE -1   // something went wrong somewhere and will terminate with error.

#define _USE_MATH_DEFINES // otherwise math constants wont work (this happens on many platforms)
#include <math.h>