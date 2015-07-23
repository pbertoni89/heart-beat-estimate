#include "ProcessBlock.h"

char *malloc_jade_err = "\nMemory error allocating structures for JADE algorithms. Will have UNATTENDED CONSEQUENCES!\n";
char *matlab_ok_plot = "\n  Data block plotted.\n";

using namespace std;
using namespace cv;

//----Constructors----------------------------------------------------------------------------------------------
#ifdef MATLAB
ProcessBlock::ProcessBlock(queue<Mat> _frameBlock, double _avgfps, bool _last, Engine *_engine) {
	engine = _engine;
#else
ProcessBlock::ProcessBlock(queue<Mat> _frameBlock, double _avgfps, bool _last) {
#endif

	avgfps = _avgfps;
	frameBlock = _frameBlock;
	last = _last;

	blockDim = frameBlock.size()-1;
	#ifdef CAMERA
		blockDim -= FRAMEDROP;
	#endif

	interpDim = (int) (ceil(blockDim*(INTERPFREQ/avgfps))); // * 256/15 = * 17

	#ifdef CAMERA
	RR        = Mat::zeros(NCHANNEL, blockDim+FRAMEDROP, TYPEMAT1CH);   // RR.depth() = CV_64F = 6
	#else
	RR        = Mat::zeros(NCHANNEL, blockDim, TYPEMAT1CH);
	#endif
	RRdtr     = Mat::zeros(NCHANNEL, blockDim, TYPEMAT1CH);
	RRdtr_std = Mat::zeros(NCHANNEL, blockDim, TYPEMAT1CH);
	RRinterp  = Mat::zeros(NCHANNEL, interpDim, TYPEMAT1CH);    // WRN: these were (rows, cols, type) Mat constructor!!!!!!!!!!!!!!1
	RRfft     = Mat::zeros(NCHANNEL, interpDim, TYPEMAT1CH);	//

	HR = new double[NCHANNEL];
}
//----Methods---------------------------------------------------------------------------------------------------
// Processes its own block of frames. Sort of "doAll()"
void ProcessBlock::process(void) {

/* II_2 */
	#ifdef DEBUGDETREND
		setDebugM(RR);
	#else
		splitChannels(frameBlock, RR);
	#endif
	#ifdef CAMERA
		RR = RR(Range(0, RR.rows), Range(FRAMEDROP, RR.cols)); // drops FRAMEDROP frames from the beginning
	#endif
	#ifdef MATLAB
		if(engine)
			matlabPlot(RR, 1);
	#endif
	cout<<" 20%";
/* II_3 */ 	
	#ifdef DETREND
		detrend(RRdtr, RR);
	#else
		RRdtr = RR;
	#endif
	normalize(RRdtr_std, RRdtr);
	#ifdef MATLAB
		if(engine) {
			matlabPlot(RRdtr, 2);
			matlabPlot(RRdtr_std, 3);
		}
	#endif
	cout<<" 40%";
/* II_4 */
	ICA_JADE(RRdtr_std);
	#ifdef MATLAB
		if(engine)
			matlabPlot(RRdtr_std, 4);
	#endif
	cout<<" 60%";
/* III_1 */
	filtSource(RRdtr_std, AVGPOINTS, BPPOINTS);
	#ifdef MATLAB 
		if(engine)
			matlabPlot(RRdtr_std, 5); 
	#endif
	cout<<" 80%";
/* III_2 */
	interpolate(RRdtr_std, RRinterp);
	#ifdef MATLAB 
		if(engine)
			matlabPlot(RRinterp, 6); 
	#endif
	cout<<" 100%";
/* III_3 */
	myFFT(RRinterp, RRfft);
	#ifdef MATLAB 
		if(engine)
			matlabPlot(RRfft, 7); 
	#endif

/* III_4 */
	findMax(RRfft);
}
//___________________________________________________________________________________________________________
// Splits a frameBlock into R,G,B channels; creates signals as difference between spatial colour means at frames i, i-1.
void ProcessBlock::splitChannels(queue<Mat> frameBlock, Mat& M) {
	
	Mat spaces[NCHANNEL];
	double mean_ch[NCHANNEL];
	double *p[NCHANNEL];
	for(int j=0; j<NCHANNEL; j++)
		p[j] = M.ptr<double>(j);
	int i = 0;
	while(frameBlock.size()) {

		split(frameBlock.front(), spaces); 

		for(int j=0; j<NCHANNEL; j++) {
			if (i)
				p[j][i-1] = (double)(mean(spaces[j]).val[0]) - mean_ch[j];
			mean_ch[j] = (double)mean(spaces[j]).val[0];
		}
		frameBlock.pop();
		i++;
	}
}
//_________________________________________________________________________________________________________________________
// Detrending close formula based on the works by Mika Tarvainen.
#ifdef DETREND
void ProcessBlock::detrend(Mat& Mdtr, Mat& M) {

	int dim = M.cols;
	double *p = NULL;
	Mat I = Mat::eye(dim, dim, TYPEMAT1CH);       // observation matrix (identity)
	Mat D2 = Mat::zeros(dim-2, dim, TYPEMAT1CH);  // two-order differential matrix

	for(int i=0; i < dim-2; i++) {
		p = D2.ptr<double>(i); 
		p[i] = 1;
		p[i+1] = -2;
		if (i < dim-1)
			p[i+2] = 1;
	}

	Mat Zstat = (I -( I -( (LAMBDA*LAMBDA) * D2.t() * D2 )).inv(DECOMP_SVD)) * M.t(); // used DECOMP_SVD before
	Mdtr = Zstat.t();

	#if HUNGRY >= 0
		if(last) {
			I.release();
			D2.release();
		}
	#endif
}
#endif
//_________________________________________________________________________________________________________________________
// For every time-series of a given set, subtracts its mean and divide it by its standard deviation.
void ProcessBlock::normalize(Mat &Mstd, Mat &M) {

	double std_dev = sqrt((double)M.cols);
	for(int i=0; i<NCHANNEL; i++) {
		Mstd.row(i) = M.row(i) - mean(M.row(i)).val[0];
		Mstd.row(i) /= norm(Mstd.row(i), NORM_L2) / std_dev;
	}
}
//_________________________________________________________________________________________________________________________
// Performs Indipendent Component Analysis using Joint Approximate Diagonalization of Eigenvalues.
#ifndef JADEMATLAB
void ProcessBlock::ICA_JADE(Mat& M) {

	int nRow = M.rows;
	int dim = M.cols;
	int m = 4;
	double treshold = 0.1 / sqrt((double)dim);

	Mat matToInv = (M * M.t()) / dim;
	Mat I = Mat::eye(nRow, nRow, TYPEMAT1CH);

	for (int i=0; i<dim; i++) { //TODO capire perchè in questo ciclo i non è usata, ma è solo un ripetitore.
		Mat Yinv = matToInv.inv();
		Mat Zinv = I.inv();
		matToInv = (matToInv + Zinv) / 2;
		I = (I + Yinv) / 2;
	}

	matToInv = matToInv.inv(DECOMP_SVD);
	M = matToInv * M;
	bool encore = true;
	Mat xi = Mat::zeros(1, dim, TYPEMAT1CH);
	Mat eta = Mat::zeros(1, dim, TYPEMAT1CH);

	while (encore) {
		encore = false;
		for (int i=0; i<nRow-1; i++) { //       i=   0,  1
			for (int j=i+1; j<nRow; j++) { //   j=   1,  2

				xi = M.row(i).mul(M.row(j)); // A.mul(B): element-wise multiplication
				eta = M.row(i).mul(M.row(i)) - M.row(j).mul(M.row(j));

				Mat arg1 = m*(eta * xi.t());
				Mat arg2 = eta * eta.t() - m*(xi * xi.t());
				double theta = atan2(arg1.at<double>(0), arg2.at<double>(0)); //atan2(0, -1)= pi

				if (fabs(theta) > treshold) {
					encore = true;
					
					Mat rowI = M.row(i);
					Mat rowJ = M.row(j);

					M.row(i) = cos(theta/m) * rowI + sin(theta/m) * rowJ;
					M.row(j) = cos(theta/m) * rowJ - sin(theta/m) * rowI;
				}
			}
		}
	}
}
#else
void ProcessBlock::ICA_JADE(Mat& M) {

	int dim = M.cols;
	int sizeRow = sizeof(double)*dim;
	mxArray *y[NCHANNEL], *x[NCHANNEL];
	double *yd[NCHANNEL], *xd[NCHANNEL];
	for(int j=0; j<NCHANNEL; j++) {
		yd[j] = (double*)malloc(sizeRow);
		for(int i=0; i<dim; i++)
			yd[j][i] = M.ptr<double>(j)[i];
		y[j] = mxCreateDoubleMatrix(1, dim, mxREAL);
		memcpy( (void*)mxGetPr(y[j]), (void*)yd[j], sizeRow);
		if(!y[j]||!yd) {
			cout<<malloc_jade_err;
			exit(ERRORSTATE); // try this
		 }
	 }
	engPutVariable(engine, "y0", y[0]); 
	engPutVariable(engine, "y1", y[1]); 
	engPutVariable(engine, "y2", y[2]);
	engEvalString(engine, "Y = [y0; y1; y2];");
	
	#if JADEMATLAB == 2
		engEvalString(engine, "[A X] = jade_1_5_fast(Y);");
	#endif
	#if JADEMATLAB == 1
		engEvalString(engine, "[A X] = jade_1_5(Y);");
	#endif
	#if JADEMATLAB == 0
		engEvalString(engine, "[A X] = jade_1_0(Y);");
	#endif
	engEvalString(engine, " x0 = X(1,:); x1 = X(2,:); x2 = X(3,:);");
	
	x[0] = engGetVariable(engine, "x0");
	x[1] = engGetVariable(engine, "x1");
	x[2] = engGetVariable(engine, "x2");
	if(!x){
		cout<<malloc_jade_err;
		exit(ERRORSTATE); // try this
	 }
	for(int j=0; j<NCHANNEL; j++) {
		xd[j] = (double*)malloc(sizeRow);
		memcpy( (void*)xd[j], (void*)mxGetPr(x[j]), sizeRow);
		for(int i=0; i<dim; i++) 
			M.ptr<double>(j)[i] = xd[j][i];
	 }
	 #if HUNGRY >= 0
		for(int j=0; j<NCHANNEL; j++) {
			mxDestroyArray(x[j]);
			mxDestroyArray(y[j]);
			free(xd[j]);
			free(yd[j]);
		}
	#endif
}
#endif
//_________________________________________________________________________________________________________________________
// Filters color signals with a avg-points moving average FIR filter plus a bp-points Hamming Window bandpass filter 
void ProcessBlock::filtSource(Mat& M, int avg_points, int bp_points){

	CV_Assert(M.depth() != sizeof(double)); //6!=8   ma perchè?

	Mat conv_kernel = Mat::ones(1, avg_points, TYPEMAT1CH)/avg_points; //vector [1/5, 1/5, 1/5, 1/5, 1/5]

	// SMOOTHING
	filter2D(M, M, -1, conv_kernel, Point(-1,-1)); 
	//negative depth will be equal to M.depth()                               
	//Point(-1,-1): center of the kernel is used as the anchor point
	//delta = 0.0 scalar will be added at end of convolution to every element 
	//border DEFAULT=4                                                      ma questi da dove vengono?

	// cfr Other\Luky\OldBp128Hamm for old hamming mask
	Mat hamming_bandpass = ( Mat_<double>(1, bp_points) <<   //al limite backspace della prima riga.
-7.1604300e-05,  -1.5032411e-04,   5.6108430e-04,   7.3410839e-04,   1.1570946e-04,   2.0700525e-04,   9.8724180e-04,   7.0422213e-04,  -2.7546869e-04,   4.4529514e-05,   7.8364615e-04,  -2.7465438e-04,  -1.5369112e-03,  -6.2448264e-04,   2.5476598e-05,  -1.8524971e-03,  -2.6811866e-03,  -4.6127152e-04,   7.6320409e-05,  -2.3209980e-03,  -1.6317669e-03,   2.2317201e-03,   1.9916318e-03,  -8.5623817e-04,   1.8548133e-03,   6.2870199e-03,   3.4911367e-03,  -1.4057886e-04,   4.2517360e-03,   7.0064398e-03,  -9.1444787e-05,  -3.9038713e-03,   2.2100184e-03,   1.6617387e-03,  -9.2110687e-03,  -1.0043925e-02,  -8.2620101e-04,  -4.6666792e-03,  -1.5979747e-02,  -9.0458784e-03,   4.1519522e-03,  -3.2523078e-03,  -1.1487899e-02,   5.8028595e-03,   1.9484773e-02,   4.9488973e-03,   7.2509616e-04,   2.6085767e-02,   3.0967262e-02,   2.9723220e-03,   3.0027089e-03,   3.1852233e-02,   1.6320318e-02,  -2.8897331e-02,  -1.7732518e-02,   1.4030016e-02,  -3.3292473e-02,  -9.2798413e-02,  -4.5058742e-02,  -1.2268463e-03,  -1.1410135e-01,  -1.9496036e-01,   3.4367834e-02,   3.8040900e-01,   3.8040900e-01,   3.4367834e-02,  -1.9496036e-01,  -1.1410135e-01,  -1.2268463e-03,  -4.5058742e-02,  -9.2798413e-02,  -3.3292473e-02,   1.4030016e-02, -1.7732518e-02,  -2.8897331e-02,   1.6320318e-02,   3.1852233e-02,   3.0027089e-03,   2.9723220e-03,   3.0967262e-02,   2.6085767e-02,   7.2509616e-04,   4.9488973e-03,   1.9484773e-02,   5.8028595e-03,  -1.1487899e-02,  -3.2523078e-03,   4.1519522e-03,  -9.0458784e-03,  -1.5979747e-02,  -4.6666792e-03,  -8.2620101e-04,  -1.0043925e-02,  -9.2110687e-03 ,  1.6617387e-03 ,  2.2100184e-03 , -3.9038713e-03 , -9.1444787e-05 ,  7.0064398e-03  , 4.2517360e-03 , -1.4057886e-04 ,  3.4911367e-03  , 6.2870199e-03  , 1.8548133e-03 , -8.5623817e-04,   1.9916318e-03 ,  2.2317201e-03,  -1.6317669e-03,  -2.3209980e-03 ,  7.6320409e-05 , -4.6127152e-04 , -2.6811866e-03 , -1.8524971e-03 ,  2.5476598e-05 , -6.2448264e-04 , -1.5369112e-03 , -2.7465438e-04 ,  7.8364615e-04 ,  4.4529514e-05 , -2.7546869e-04 ,  7.0422213e-04 ,  9.8724180e-04,   2.0700525e-04 ,  1.1570946e-04 ,  7.3410839e-04 ,  5.6108430e-04,  -1.5032411e-04 , -7.1604300e-05);

	// FILTERING
	filter2D(M, M, -1, hamming_bandpass, Point(-1,-1));
}
//_________________________________________________________________________________________________________________________
//To refine the BVP peak fiducial point, the signal was interpolated with a cubic spline function.
void ProcessBlock::interpolate(Mat &Msrc, Mat &Mdst) {

	/* Usually fx(x,y) and fy(x,y) are floating-point numbers, 
	   so a pixel values at fractional coordinates needs to be retrieved. 
	   In the simplest case the coordinates can be just rounded to the nearest integer coordinates 
	   and the corresponding pixel used, which is called "nearest-neighbor" interpolation. 
	   However, a better result can be achieved by using more sophisticated interpolation methods, 
	   where a polynomial function is fit into some neighborhood of the computed pixel ( fx(x,y), fy(x,y) )
	   and then the value of the polynomial at the same pixel is taken as the interpolated pixel value. */	

	Size sz_interp = Size(Mdst.cols, Mdst.rows);
	double x_scale = INTERPFREQ/avgfps; //The scale factor along the horizontal axis. with 256/15 = 17.06
	double y_scale = 0;           //by default it is computed as (double)(sz_interp).height/src.rows = Mdst.rows/Msrc.rows = NCHANNEL/NCHANNEL = 1
	// with similar values, just horizontal (on frames) interpolation will be applied.
	cv::resize(Msrc, Mdst, sz_interp, x_scale, y_scale, INTER_CUBIC);
}
//_________________________________________________________________________________________________________________________
// Returns the set of amplitude specters of a given set of time signals.
void ProcessBlock::myFFT(Mat &Minterp, Mat &Mfft) {

	int c = Minterp.cols, r = Minterp.rows;
	fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * c);
	fftw_complex* in  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * c);
	fftw_plan dft_plan;

	double *p;
	for (int j=0; j<r; j++) {
		p = Minterp.ptr<double>(j);

		for (int k=0; k<c; k++) {
			in[k][0] = p[k]; // real part
			in[k][1] = 0;    // imag part
		}

		dft_plan = fftw_plan_dft_1d(c, in, out, FFTW_FORWARD,FFTW_ESTIMATE); //takes N=c: so no interpolation rises (interpolated yet)
		fftw_execute(dft_plan);                                              //ESTIMATE do not turn on benchmark & planning: if you need em, use MEASURE

		p = Mfft.ptr<double>(j);
		for (int i = 0; i<c; i++)//do modulus
			p[i] = sqrt(pow(out[i][0], 2) + pow(out[i][1], 2));
	}

	#if HUNGRY >= 1
		if(last) {
			fftw_destroy_plan(dft_plan);
			fftw_free(in);
			fftw_free(out);
		}
	#endif
}
//_________________________________________________________________________________________________________________________
void ProcessBlock::findMax(Mat& Mfft){

	double max[NCHANNEL];
	int index[NCHANNEL];  // Array of indexes (they are frequencies) of maximum points found.
	int dim = Mfft.cols;
	double tmp;

	for(int j=0; j<NCHANNEL; j++)
		max[j] = Mfft.ptr<double>(j)[0];

	for (int i = 1; i < dim/2; i++) // since we are taking the positive half of FFT (stored into first falf of Mfft.ptr(xx)
		for(int j=0; j<NCHANNEL; j++) {
			tmp = Mfft.ptr<double>(j)[i];
			if (max[j] < tmp) {
				max[j] = tmp;
				index[j] = i;
			}
		}
		
	for(int j=0; j<NCHANNEL; j++)
		 HR[j] = (double)(index[j]*INTERPFREQ*60)/dim;
}
//_________________________________________________________________________________________________________________________
#ifdef MATLAB
// Plots channels of an array of cv::Mat as B,G,R time signals.
void ProcessBlock::matlabPlot(Mat toPlot, int flags) {
	
	int dim = toPlot.cols;
	int correct = 20; // Since positive half of FFT is up to 128 Hz, we are interested into showing only the first 5 Hz = 120/20
	if(flags==7) 
		dim /= (2*correct);        // cause we are representing only the positive half of FFT.
	int sizeRow = sizeof(double)*dim;
		
	mxArray *ax = NULL;                                 // Frame axis of interest.
	mxArray *ch[NCHANNEL];                       // Values of different channels.
	double *xd = (double*)malloc(sizeRow);		// array of c doubles
	double *yd[NCHANNEL];							// (NCHANNEL x c) matrix of doubles

	double scale;
	if(flags<=5)
		#ifndef DEBUGDETREND
			scale = avgfps;
		#else
			scale = 1/DEBUGDT;
		#endif
	else if(flags==6)
		scale = INTERPFREQ;
	else
		scale = INTERPFREQ/2;  // cause we are representing only the positive half of FFT.

	for(int j=0; j<NCHANNEL; j++)
		yd[j] = (double*)malloc(sizeRow);

	if(flags!=7)
		for(int i=0; i<dim; i++) { // fixing x scale
			xd[i] = i*(1/scale);
			for(int j=0; j<NCHANNEL; j++)
				yd[j][i] = toPlot.ptr<double>(j)[i]; 
		}
	else
		for(int i=0; i<dim; i++) {
			xd[i] = (double)(scale/(dim*correct))*(i); 
			for(int j=0; j<NCHANNEL; j++)
				yd[j][i] = toPlot.ptr<double>(j)[i]; 
		}

	for(int j=0; j<NCHANNEL; j++) {
		ch[j] = mxCreateDoubleMatrix(1, dim, mxREAL);
		memcpy( (void*)mxGetPr(ch[j]), (void*)yd[j], sizeRow);
	 }

	engPutVariable(engine, "ch0", ch[0]);
	engPutVariable(engine, "ch1", ch[1]);
	engPutVariable(engine, "ch2", ch[2]);

	//cannot shift these three lines into case 1 (as in v0.1 - v0.2), since after interpolation #frames simply changes.
	ax = mxCreateDoubleMatrix(1, dim, mxREAL);
	memcpy( (void*)mxGetPr(ax), (void*)xd, sizeRow);
	engPutVariable(engine, "ax", ax);
		
	switch(flags) {
	case 1: //RR
		engEvalString(engine, "figure('name','RR series as difference of R series'); clf;");
		engEvalString(engine, "subplot(3,1,1); plot(ax,ch0, 'b'); axis tight;");
		engEvalString(engine, "subplot(3,1,2); plot(ax,ch1, 'g'); axis tight; xlabel('seconds');" );
		engEvalString(engine, "subplot(3,1,3); plot(ax,ch2, 'r'); axis tight;");
	break; 
	case 2: //RRdtr
		#ifdef DEBUGDETREND
		engEvalString(engine, "subplot(3,1,1); hold on; plot(ax,ch0, 'k--'); axis tight");
		engEvalString(engine, "subplot(3,1,2); hold on; plot(ax,ch1, 'k--'); axis tight");
		engEvalString(engine, "subplot(3,1,3); hold on; plot(ax,ch2, 'k--'); axis tight");
		#else
		engEvalString(engine, "figure('name','Detrending & Normalizing RR series'); clf;");
		engEvalString(engine, "subplot(3,1,1); plot(ax,ch0, 'b'); axis tight;");
		engEvalString(engine, "subplot(3,1,2); plot(ax,ch1, 'g'); axis tight; xlabel('seconds');");
		engEvalString(engine, "subplot(3,1,3); plot(ax,ch2, 'r'); axis tight;");
		#endif
	break;
	case 3:
		#ifndef DEBUGDETREND
		engEvalString(engine, "subplot(3,1,1); hold on; plot(ax,ch0, 'k--'); axis tight;");
		engEvalString(engine, "subplot(3,1,2); hold on; plot(ax,ch1, 'k--'); axis tight;");
		engEvalString(engine, "subplot(3,1,3); hold on; plot(ax,ch2, 'k--'); axis tight; hold off;");
		#endif
	break;
	case 4: //ICA_JADE 
		engEvalString(engine, "figure('name','Indipendent Component Analysis through Joint Approximate Diagonalization of Eigenmatrices'); clf;"); 
		engEvalString(engine, "subplot(3,1,1); plot(ax,ch0, 'k'); axis tight;");
		engEvalString(engine, "subplot(3,1,2); plot(ax,ch1, 'k'); axis tight; xlabel('seconds');");
		engEvalString(engine, "subplot(3,1,3); plot(ax,ch2, 'k'); axis tight;");
	break;
	case 5: //filtSource
		engEvalString(engine, "subplot(3,1,1); hold on; plot(ax,ch0, 'r--'); axis tight;");
		engEvalString(engine, "subplot(3,1,2); hold on; plot(ax,ch1, 'r--'); axis tight;");
		engEvalString(engine, "subplot(3,1,3); hold on; plot(ax,ch2, 'r--'); axis tight; hold off;");
	break;	
	case 6: //interp
		engEvalString(engine, "figure('name','Interpolated BVP Signal obtained from filtering ICA signals'); clf;"); 
		engEvalString(engine, "subplot(3,1,1); plot(ax,ch0, 'b'); axis tight;");
		engEvalString(engine, "subplot(3,1,2); plot(ax,ch1, 'b'); axis tight; xlabel('seconds');");
		engEvalString(engine, "subplot(3,1,3); plot(ax,ch2, 'b'); axis tight;");
	break;
	case 7: //amplitudes
		engEvalString(engine, "figure('name','Amplitude Specters of Relevant BVP signals'); clf;"); 
		engEvalString(engine, "subplot(3,1,1); plot(ax,ch0, 'k'); axis tight;");
		engEvalString(engine, "subplot(3,1,2); plot(ax,ch1, 'k'); axis tight; xlabel('hertz');");
		engEvalString(engine, "subplot(3,1,3); plot(ax,ch2, 'k'); axis tight;");
	}

	#if HUNGRY >= 0
		engEvalString(engine, "clear;");
		mxDestroyArray(ax);
		free(xd);
		for(int j=0; j<NCHANNEL; j++) {
			mxDestroyArray(ch[j]);
			free(yd[j]);
		}
	#endif
}
#endif
//_________________________________________________________________________________________________________________________
double* ProcessBlock::getHR(void) {
	return HR;
}
//----Destructors----------------------------------------------------------------------------------------------------------
ProcessBlock::~ProcessBlock(void) {
	#if HUNGRY >= 0
		RR.release();
		RRdtr.release();
		RRdtr_std.release();
		RRinterp.release();
		RRfft.release();
	#endif
}
//-------------------------------------------------------------------------------------------------------------------------
//		Debug Functions
//-------------------------------------------------------------------------------------------------------------------------
#ifdef DEBUGDETREND
// Creates known signals to observe detrending algorithm. Actually SIMULATES 50 seconds of video, because 50 = DEBUGDIM*DEBUGDT
void ProcessBlock::setDebugM(Mat &M) {

	double di;
	double *p0 = M.ptr<double>(0), *p1 = M.ptr<double>(1), *p2 = M.ptr<double>(2);
	double f0a = 0.25, f0b = 0.3333333333, f1 = 0.125, f2 = 0.0666666667; // periods are 4, 3, 8, 15 
	double m0 = 0.2, m1 = 0.4, m2 = 0.35;
	int A0 = 2, B0 = 8, A1 =3, A2 = 2;
	for(int i=0; i<M.cols; i++) { 
		di = i*DEBUGDT;
		p0[i] =  m0*di  + A0*cos(di*f0a*2*M_PI) + B0*cos(di*f0b*2*M_PI);
		p1[i] =  m1*di  + A1*sin(di*f1 *2*M_PI);
		p2[i] = -m2*di  + A2*sin(di*f2 *2*M_PI);
	} 
}
#endif