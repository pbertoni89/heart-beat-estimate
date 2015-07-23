#pragma once

#include "common_cfg.h"
#include <fftw3.h>

#define NCHANNEL 3         // R,G,B color channels.
#define AVGPOINTS 5        // points for moving average smoothing
#define BPPOINTS 128       // points for bandpass hamming filtering
#define INTERPFREQ 256     // sampling frequency of cubic spline interpolation. Choice based on confront with Medical equipment frequency rate.
#define TYPEMAT1CH CV_64FC1// 32/64 bit depth ; F(loat) | D(ouble) | U(nsigned ; C(hannels) 1,2,3.. ; row0 = B, row1 = G, row2 = R
#define TYPEMAT3CH CV_64FC3

#ifdef MATLAB
	#define JADEMATLAB 2   // allows to call third part Matlab functions to perform ICA_JADE algorithm. 
						   // EXTREME CARE: these files MUST BE in your Matlab Path, or calls will simply fail!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
						   // At the moment we are not handling these exceptions, so the outputs will be uncertain (probably SEGFAULT)
						   // 0 algorithm v1.0
						   // 1 algorithm v1.5
						   // 2 algorithm v1.5fast: should require less CPU computations but more RAM space.
						   // #undefine to use old & deprecated self-written JADE algorithm.
#endif
#define DETREND			   // Turns on Detrending Algorithm from Mika Tarvainen. At the moment it is working wrong, so we are excluding it.
#ifdef DETREND
	#define LAMBDA 1000000000 // exponential order of smoothing parameter in detrending algorithm. Test with 10.
	#define DEBUGDETREND   // Uses 3 given signals to test Detrending Algorithm. #undefine to use video capture instead (default)
	//#define DEBUGDIM 500
	#define DEBUGDT 0.1
#endif

using namespace cv;
using namespace std;

class ProcessBlock {

	//----Variables--------------------------------------------------------------------------------------------------------
	queue<Mat> frameBlock;
	#ifdef MATLAB
		Engine *engine;
	#endif
	bool last;
	int blockDim;
	int interpDim;

	double avgfps; // real fps while preprocessing frames.
	double *HR;

	Mat RR;  // vector signal with on every row a different color; and on every column the difference between
	Mat RRdtr;
	Mat RRdtr_std;
	Mat RRinterp;
	Mat RRfft;
	
public:
	//----Constructors-----------------------------------------------------------------------------------------------------
#ifdef MATLAB
	ProcessBlock(queue<Mat> _frameBlock, double _realfps, bool _last, Engine *_engine);
#else
	ProcessBlock(queue<Mat> _frameBlock, double _realfps, bool _last);
#endif
	//----Destructor------------------------------------------------------------------------------------------------------
	~ProcessBlock(void);
	//----Methods----------------------------------------------------------------------------------------------------------
	void process(void);
	double* getHR(void);
private:
	void splitChannels(queue<Mat> frameBlock, Mat& M);
	#ifdef CAMERA
		void dropStart(Mat& M, Mat& Mclean);
	#endif
	void detrend(Mat &Mdtr, Mat &M);
	void normalize(Mat &Mstd, Mat &M);
	void ICA_JADE(Mat &M);
	void filtSource(Mat &M, int avg_points, int bp_points);
	void interpolate(Mat &Msrc, Mat &Mdst);
	void myFFT(Mat &Minterp, Mat &Mfft);
	void findMax(Mat &Mfft);
	#ifdef MATLAB
		void matlabPlot(Mat toPlot, int flags);
	#endif
	#ifdef DEBUGDETREND
		void setDebugM(Mat &M);
	#endif
};



