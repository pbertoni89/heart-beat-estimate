#include <highgui.h>
#include <cv.h>
#include "cv.h"
#include <stdio.h>
#include <math.h>
#include <fftw3.h>

using namespace std;
using namespace cv;

#define NFRAME 10
#define SMOOTH 10
#define FRAMEH 480
#define FRAMEW 640
#define FPS 15
#define INTFREQ 256

String face_cascade_name = "haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;

void ICA_JADE(Mat& RR);
void filtSource(Mat& RR, int point);
void doubleMax(Mat& RRfft);

int main() {

	VideoCapture cap(0); // open the default camera
	//VideoCapture cap1("test.avi");

	cap.set(CV_CAP_PROP_FRAME_WIDTH, FRAMEW);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, FRAMEH);
	cap.set(CV_CAP_PROP_FPS, FPS);
//	cap1.set(CV_CAP_PROP_FRAME_WIDTH, FRAMEW);
//	cap1.set(CV_CAP_PROP_FRAME_HEIGHT, FRAMEH);
//	cap1.set(CV_CAP_PROP_FPS, FPS);
//	cap.set(CV_CAP_PROP_CONVERT_RGB,true);

	if (!cap.isOpened())
		return -1;

	Mat frame;
	cap >> frame;

	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	if (!face_cascade.load(face_cascade_name)) {
		return -1;
	}

	vector<Rect> faces;
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 3, 0, cvSize(1, 1),cvSize(5, 5));

	faces[0].x = faces[0].x + 0.2 * faces[0].width;
	faces[0].y = faces[0].y + 0.1 * faces[0].height;
	faces[0].height = 0.8 * faces[0].height;
	faces[0].width = 0.6 * faces[0].width;

	frame_gray.release();

	Mat spaces[3];
	Mat RR = Mat::zeros(frame.channels(), NFRAME - 1, CV_64FC1);
	double* pRRB = RR.ptr<double>(0);
	double* pRRG = RR.ptr<double>(1);
	double* pRRR = RR.ptr<double>(2);
	double meanR, meanB, meanG;

	namedWindow("Capture");

	//frame=Mat::ones(FRAMEW, FRAMEH, CV_64FC3)*128;


	for (int i = 0; i < NFRAME; i++) {
		cap >> frame;
		split(frame(faces[0]), spaces);
		if (i != 0) {
			pRRB[i - 1] = (double)mean(spaces[0]).val[0] - meanB;
			pRRG[i - 1] = (double)mean(spaces[1]).val[0] - meanG;
			pRRR[i - 1] = (double)mean(spaces[2]).val[0] - meanR;
		}
		meanB = (double)mean(spaces[0]).val[0];
		meanG = (double)mean(spaces[1]).val[0];
		meanR = (double)mean(spaces[2]).val[0];

		//cout<<(double)mean(spaces[0]).val[0]<<" "<<(double)mean(spaces[1]).val[0]<<" "<<(double)mean(spaces[2]).val[0]<<endl<<endl;
		//cout<<pRRB[i - 1]<<" "<<pRRG[i - 1]<<" "<<pRRR[i - 1]<<endl<<endl;

		imshow("Capture",frame(faces[0]));
	}

	cap.release();

	double* p;
	Mat I = Mat::eye(NFRAME - 1, NFRAME - 1, CV_64FC1);
	Mat D2 = Mat::zeros(NFRAME - 1, NFRAME - 1, CV_64FC1) + I;
	for (int i = 0; i < NFRAME - 2; i++) {
		p = D2.ptr<double>(i);
		p[i + 1] -= 2;
		if (i < NFRAME - 3) {
			p[i + 2] += 1;
		}
	}

	Mat RRmod(3,NFRAME-1,CV_64FC1);

	RRmod = (I-(I-((SMOOTH^2)*D2.t()*D2)).inv(DECOMP_SVD))*RR.t();

	RRmod = RRmod.t();

	//cout<<RRmod<<endl<<endl;

	RR.release();
	I.release();
	D2.release();

	//cout<<mean(RRmod.row(0)).val[0]<<" "<<mean(RRmod.row(1)).val[0]<<" "<<mean(RRmod.row(2)).val[0]<<endl<<endl;

	RRmod.row(0) -= mean(RRmod.row(0)).val[0];
	RRmod.row(1) -= mean(RRmod.row(1)).val[0];
	RRmod.row(2) -= mean(RRmod.row(2)).val[0];

	RRmod.row(0) /= norm(RRmod.row(0), NORM_L2) / sqrt(NFRAME - 1);
	RRmod.row(1) /= norm(RRmod.row(1), NORM_L2) / sqrt(NFRAME - 1);
	RRmod.row(2) /= norm(RRmod.row(2), NORM_L2) / sqrt(NFRAME - 1);

	ICA_JADE(RRmod);

	filtSource(RRmod, 5);

	Mat RRinterp(RRmod.rows, RRmod.cols * (int) (INTFREQ / FPS), CV_64FC1);

	resize(RRmod, RRinterp, Size(RRinterp.cols, RRinterp.rows),double(INTFREQ / FPS), 0, INTER_CUBIC);

	RRmod.release();

	Mat RRfft(RRinterp.rows, RRinterp.cols, CV_64FC1);

	fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * RRinterp.cols);
	fftw_complex* in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * RRinterp.cols);
	fftw_plan pl;

	for (int j = 0; j < RRinterp.rows; j++) {
		p = RRinterp.ptr<double>(j);

		for (int k = 0; k < RRinterp.cols; k++) {
			in[k][0] = p[k];
			in[k][1] = 0;
		}


		pl = fftw_plan_dft_1d(RRinterp.cols, in, out, FFTW_FORWARD,FFTW_ESTIMATE);
		fftw_execute(pl);

		p = RRfft.ptr<double>(j);
		for (int k = 0; k < RRinterp.cols; k++) {
			p[k] = sqrt(pow(out[k][0], 2) + pow(out[k][1], 2));
		}
	}

	RRinterp.release();

	fftw_destroy_plan(pl);
	fftw_free(in);
	fftw_free(out);

	doubleMax(RRfft);

	RRfft.release();

	return 0;
}

void ICA_JADE(Mat& RR) {

	int nRow = RR.rows;
	int nCol = RR.cols;

	Mat MatToInv = (RR * RR.t()) / nCol;
	Mat I = Mat::eye(nRow, nRow, CV_64FC1);

	for (int i = 0; i < 100; i++) {
		Mat Yinv = MatToInv.inv();
		Mat Zinv = I.inv();
		MatToInv = (MatToInv + Zinv) / 2;
		I = (I + Yinv) / 2;
	}

	MatToInv = MatToInv.inv(DECOMP_SVD);
	RR = MatToInv * RR;

	int encore = 1;
	double c, s;
	Mat xi = Mat::zeros(1, nCol, CV_64FC1);
	Mat eta = Mat::zeros(1, nCol, CV_64FC1);

	while (encore) {
		encore = 0;
		for (int i = 0; i < nRow - 1; i++) {
			for (int j = i + 1; j < nRow; j++) {

				xi = RR.row(i).mul(RR.row(j));
				eta = RR.row(i).mul(RR.row(i)) - RR.row(j).mul(RR.row(j));

				Mat arg1 = 4 * (eta * xi.t());
				Mat arg2 = eta * eta.t() - 4 * (xi * xi.t());
				double omega = atan2(arg1.at<double>(0), arg2.at<double>(0));

				if (fabs(omega) > (double) 0.1 / sqrt(nCol)) {

					encore = 1;

					c = cos(omega / 4);
					s = sin(omega / 4);

					Mat rowI = RR.row(i);
					Mat rowJ = RR.row(j);

					RR.row(i) = c * rowI + s * rowJ;
					RR.row(j) = c * rowJ - s * rowI;
				}
			}
		}
	}
}

void filtSource(Mat& RR, int point){

	CV_Assert(RR.depth() != sizeof(double));

	Mat kernel = Mat::ones(1,point,CV_64FC1)/point;
	Point zero(-1,-1);

	filter2D(RR,RR,-1,kernel,zero);

	//Mat bp128hamm = (Mat_<double>(1,128)<< 0.000409242431351856, 0.000709191409703230, 0.000791094121466546, 0.000604101403079529, 0.000267233047559045, 1.63832294442686e-05, 6.93186252409334e-05, 0.000475978441212351, 0.00104997200397036, 0.00144924111730749, 0.00138691962984501, 0.000852347828569683, 0.000181218498618594, -0.000115510829788772, 0.000305673605943316, 0.00132454207778391, 0.00234254492310823, 0.00262500686338201, 0.00181977667734338, 0.000298027132425864, -0.000977293407242571, -0.00105268069769898, 0.000303166924931752, 0.00228724547292114, 0.00345980115405875, 0.00268695957528940, 2.74406599136683e-05, -0.00310270327371272, -0.00473407249300086, -0.00368982044105150, -0.000576456849581081, 0.00236362405958739, 0.00261391018198149, -0.000838361795371643, -0.00653373617946487, -0.0112079664272943, -0.0118631979393323, -0.00796242505247725, -0.00218750919663162, 0.000997301195088242, -0.00172672923894365, -0.00998487461129213, -0.0194134499246455, -0.0241597333963824, -0.0208173430324981, -0.0112246179949089, -0.00197913975647901, -0.000512580385000177, -0.00995407068534988, -0.0261879715159676, -0.0395054961670230, -0.0404490563205753, -0.0265131444036056, -0.00525766310756844, 0.00890530002620991, 0.00322811718200646, -0.0242010973279584, -0.0604146516278096, -0.0820554714667467, -0.0670222932121859, -0.00740364309758718, 0.0835945998193629, 0.175282351413949, 0.232644363670543, 0.232644363670543, 0.175282351413949, 0.0835945998193629, -0.00740364309758718, -0.0670222932121859, -0.0820554714667467, -0.0604146516278096, -0.0242010973279584, 0.00322811718200646, 0.00890530002620991, -0.00525766310756844, -0.0265131444036056, -0.0404490563205753, -0.0395054961670230, -0.0261879715159676, -0.00995407068534988, -0.000512580385000177, -0.00197913975647901, -0.0112246179949089, -0.0208173430324981, -0.0241597333963824, -0.0194134499246455, -0.00998487461129213, -0.00172672923894365, 0.000997301195088242, -0.00218750919663162, -0.00796242505247725, -0.0118631979393323, -0.0112079664272943, -0.00653373617946487, -0.000838361795371643, 0.00261391018198149, 0.00236362405958739, -0.000576456849581081, -0.00368982044105150, -0.00473407249300086, -0.00310270327371272, 2.74406599136683e-05, 0.00268695957528940, 0.00345980115405875, 0.00228724547292114, 0.000303166924931752, -0.00105268069769898, -0.000977293407242571, 0.000298027132425864, 0.00181977667734338, 0.00262500686338201, 0.00234254492310823, 0.00132454207778391, 0.000305673605943316, -0.000115510829788772, 0.000181218498618594, 0.000852347828569683, 0.00138691962984501, 0.00144924111730749, 0.00104997200397036, 0.000475978441212351, 6.93186252409334e-05, 1.63832294442686e-05, 0.000267233047559045, 0.000604101403079529, 0.000791094121466546, 0.000709191409703230, 0.000409242431351856);
	Mat bp128hamm = (Mat_<double>(1,128)<<   -7.1604300e-05,  -1.5032411e-04,   5.6108430e-04,   7.3410839e-04,   1.1570946e-04,   2.0700525e-04,   9.8724180e-04,   7.0422213e-04,  -2.7546869e-04,   4.4529514e-05,   7.8364615e-04,  -2.7465438e-04,  -1.5369112e-03,  -6.2448264e-04,   2.5476598e-05,  -1.8524971e-03,  -2.6811866e-03,  -4.6127152e-04,   7.6320409e-05,  -2.3209980e-03,  -1.6317669e-03,   2.2317201e-03,   1.9916318e-03,  -8.5623817e-04,   1.8548133e-03,   6.2870199e-03,   3.4911367e-03,  -1.4057886e-04,   4.2517360e-03,   7.0064398e-03,  -9.1444787e-05,  -3.9038713e-03,   2.2100184e-03,   1.6617387e-03,  -9.2110687e-03,  -1.0043925e-02,  -8.2620101e-04,  -4.6666792e-03,  -1.5979747e-02,  -9.0458784e-03,   4.1519522e-03,  -3.2523078e-03,  -1.1487899e-02,   5.8028595e-03,   1.9484773e-02,   4.9488973e-03,   7.2509616e-04,   2.6085767e-02,   3.0967262e-02,   2.9723220e-03,   3.0027089e-03,   3.1852233e-02,   1.6320318e-02,  -2.8897331e-02,  -1.7732518e-02,   1.4030016e-02,  -3.3292473e-02,  -9.2798413e-02,  -4.5058742e-02,  -1.2268463e-03,  -1.1410135e-01,  -1.9496036e-01,   3.4367834e-02,   3.8040900e-01,   3.8040900e-01,   3.4367834e-02,  -1.9496036e-01,  -1.1410135e-01,  -1.2268463e-03,  -4.5058742e-02,  -9.2798413e-02,  -3.3292473e-02,   1.4030016e-02, -1.7732518e-02,  -2.8897331e-02,   1.6320318e-02,   3.1852233e-02,   3.0027089e-03,   2.9723220e-03,   3.0967262e-02,   2.6085767e-02,   7.2509616e-04,   4.9488973e-03,   1.9484773e-02,   5.8028595e-03,  -1.1487899e-02,  -3.2523078e-03,   4.1519522e-03,  -9.0458784e-03,  -1.5979747e-02,  -4.6666792e-03,  -8.2620101e-04,  -1.0043925e-02,  -9.2110687e-03 ,  1.6617387e-03 ,  2.2100184e-03 , -3.9038713e-03 , -9.1444787e-05 ,  7.0064398e-03  , 4.2517360e-03 , -1.4057886e-04 ,  3.4911367e-03  , 6.2870199e-03  , 1.8548133e-03 , -8.5623817e-04,   1.9916318e-03 ,  2.2317201e-03,  -1.6317669e-03,  -2.3209980e-03 ,  7.6320409e-05 , -4.6127152e-04 , -2.6811866e-03 , -1.8524971e-03 ,  2.5476598e-05 , -6.2448264e-04 , -1.5369112e-03 , -2.7465438e-04 ,  7.8364615e-04 ,  4.4529514e-05 , -2.7546869e-04 ,  7.0422213e-04 ,  9.8724180e-04,   2.0700525e-04 ,  1.1570946e-04 ,  7.3410839e-04 ,  5.6108430e-04,  -1.5032411e-04 , -7.1604300e-05);

	filter2D(RR,RR,-1,bp128hamm,zero);
}

void doubleMax(Mat& RRfft){

	double maxR,maxG,maxB;
	int pointerR,pointerG,pointerB;

	double* pB = RRfft.ptr<double>(0);
	double* pG = RRfft.ptr<double>(1);
	double* pR = RRfft.ptr<double>(2);


	for (int i = 0; i < (RRfft.cols)/2; i++){
		if (i == 0){
			maxB=pB[i];
			maxG=pG[i];
			maxR=pR[i];
		}

		if (maxB < pB[i]){
			maxB = pB[i];
			pointerB = i;
		}

		if (maxG < pG[i]){
			maxG = pG[i];
			pointerG = i;
		}
		if (maxR < pR[i]){
			maxR = pR[i];
			pointerR = i;
		}
	}

	cout<<"The Heart Rate is (Channel B): "<< (double)((pointerB)*INTFREQ*60)/RRfft.cols<<" bpm [beats per minute]"<<endl;
	cout<<"The Heart Rate is (Channel G): "<< (double)((pointerG)*INTFREQ*60)/RRfft.cols<<" bpm [beats per minute]"<<endl;
	cout<<"The Heart Rate is (Channel R): "<< (double)((pointerR)*INTFREQ*60)/RRfft.cols<<" bpm [beats per minute]"<<endl<<endl;
}
