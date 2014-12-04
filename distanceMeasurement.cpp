#include <iostream>
#include <opencv2/opencv.hpp>

#define SCALE 2.0
#define FOCUS 545.0 //カメラの焦点距離(mm)
#define CAMERA 100.0 //Webカメラ同士の距離(mm)

int main(int argc, char *argv[])
{
	double width  =	800;
	double height =	600;
	double distance;

	cv::Mat leftImage, rightImage, calibratedLeftImage, calibratedRightImage, grayLeft, grayRight;

	// カメラからのビデオキャプチャを初期化する
	cv::VideoCapture leftFrame(0);
	cv::VideoCapture rightFrame(1);

	//キャプチャ画像をRGBで取得
	leftFrame.set( CV_CAP_PROP_FRAME_WIDTH, width );
	leftFrame.set( CV_CAP_PROP_FRAME_HEIGHT, height );
	rightFrame.set( CV_CAP_PROP_FRAME_WIDTH, width );
	rightFrame.set( CV_CAP_PROP_FRAME_HEIGHT, height );

	// ウィンドウを作成する
	char leftWindow[] = "left";
	char rightWindow[] = "right";
	cv::namedWindow( leftWindow, CV_WINDOW_AUTOSIZE );
	cv::namedWindow( rightWindow, CV_WINDOW_AUTOSIZE );

	// 分類器の読み込み(2種類あるから好きな方を)
	std::string cascadeName = "haarcascade_frontalface_default.xml";
	cv::CascadeClassifier cascade;
	if(!cascade.load(cascadeName))
	{
		std::cout << "can't find the casdade" << std::endl;
		return -1;
	}
//ここからstereoCalibrationの関数
	const int								numberOfCheckerPatterns = 5; // チェッカー画像の枚数

	cv::vector<cv::Mat>						checkerImgs1, checkerImgs2;	// チェッカーパターン画像
	
	cv::Size								imageSize;
	const cv::Size							patternSize( 10, 7 );						// チェッカーパターンの交点の数
	cv::vector<cv::vector<cv::Point3f> >	worldPoints( numberOfCheckerPatterns );		// チェッカー交点座標と対応する世界座標の値を格納する行列
	cv::vector<cv::vector<cv::Point2f> >	imagePoints1( numberOfCheckerPatterns );	// チェッカー交点座標を格納する行列
	cv::vector<cv::vector<cv::Point2f> >	imagePoints2( numberOfCheckerPatterns );	// チェッカー交点座標を格納する行列
	cv::TermCriteria						criteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.001 );	// 対応するワールド座標系パターン

	// カメラパラメータ行列
	cv::Mat			cameraMatrix1, cameraMatrix2;		// 内部パラメータ行列
	cv::Mat			distCoeffs1, distCoeffs2;			// レンズ歪み行列
	cv::Mat			R, T, E, F, R1, R2, P1, P2, Q;		// 2眼の間の関係を保持する行列群
	cv::Mat 		mapLX, mapLY, mapRX, mapRY;
	cv::Mat 		calibratedImage1, calibratedImage2;

	cv::Mat			lefttest = cv::imread( "left.jpg", 0 );
	cv::Mat			righttest = cv::imread( "right.jpg", 0 );


	// 世界座標を決める
	for( int i = 0; i < numberOfCheckerPatterns; i++ ) {
		for( int j = 0 ; j < patternSize.area(); j++ ) {
			worldPoints[i].push_back( cv::Point3f(	static_cast<float>( j % patternSize.width * 10 ), 
													static_cast<float>( j / patternSize.width * 10 ), 
													0.0 ) );
		}
	}

	// 2眼分のチェッカーパターン画像を読み込む（グレーで）
	for( int i = 0; i < numberOfCheckerPatterns; i++ ) {
		std::stringstream		stream1, stream2;
		stream1 <<  i + 1 << "l.jpg";
		stream2 <<  i + 1 << "r.jpg";
		std::string	fileName1 = stream1.str();
		std::string	fileName2 = stream2.str();
		checkerImgs1.push_back( cv::imread( fileName1, 0 ) );
		checkerImgs2.push_back( cv::imread( fileName2, 0 ) );
	}

	// 画像サイズを得る
	imageSize = cv::Size( checkerImgs1[0].cols, checkerImgs1[0].rows );

	// チェックパターンの交点座標を求め，imagePointsに格納する
	for( int i = 0; i < numberOfCheckerPatterns; i++ ) {
		std::cout << "Find corners from image " << i + 1;
		if( cv::findChessboardCorners( checkerImgs1[i], patternSize, imagePoints1[i] ) && 
			cv::findChessboardCorners( checkerImgs2[i], patternSize, imagePoints2[i] ) ) {
	
			std::cout << " ... All corners found." << std::endl;

			cv::cornerSubPix( checkerImgs1[i], imagePoints1[i], cv::Size( 11, 11 ), cv::Size( -1, -1 ), criteria );
			cv::cornerSubPix( checkerImgs2[i], imagePoints2[i], cv::Size( 11, 11 ), cv::Size( -1, -1 ), criteria );
		} else {
			std::cout << " ... at least 1 corner not found." << std::endl;
			cv::waitKey( 0 );
			return -1;
		}
	}

	// 2眼ステレオカメラ群を同時にキャリブレーション
	cv::stereoCalibrate(	worldPoints, 
							imagePoints1, imagePoints2, 
							cameraMatrix1, distCoeffs1, 
							cameraMatrix2, distCoeffs2, 
							imageSize, R, T, E, F );



	// Rectification
	cv::stereoRectify(	cameraMatrix1, distCoeffs1, 
						cameraMatrix1, distCoeffs1, 
						imageSize, R, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1,imageSize );


	cv::initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, imageSize, CV_32FC1, mapLX, mapLY);
	cv::initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, imageSize, CV_32FC1, mapRX, mapRY);



	cv::remap(lefttest, calibratedImage1, mapLX, mapLY, CV_INTER_CUBIC);
	cv::remap(righttest, calibratedImage2, mapRX, mapRY, CV_INTER_CUBIC);

	cv::namedWindow("checkerImgs1", CV_WINDOW_AUTOSIZE);
	cv::namedWindow("checkerImgs2", CV_WINDOW_AUTOSIZE);

	cv::imshow("checkerImgs1", checkerImgs1[1]);
	cv::imshow("checkerImgs2", checkerImgs2[1]);

	cv::namedWindow("calibratedImage1", CV_WINDOW_AUTOSIZE);
	cv::namedWindow("calibratedImage2", CV_WINDOW_AUTOSIZE);

	cv::imshow("calibratedImage1", calibratedImage1);
	cv::imshow("calibratedImage2", calibratedImage2);
	
	// 何かキーが押下されるまで、ループをくり返す
	while( cvWaitKey( 1 ) == -1 )
	{
		leftFrame >> leftImage;
		rightFrame >> rightImage;

		cv::remap(leftImage, calibratedLeftImage, mapLX, mapLY, CV_INTER_CUBIC);
		cv::remap(rightImage, calibratedRightImage, mapRX, mapRY, CV_INTER_CUBIC);

		// グレースケール画像に変換
		cv::cvtColor(calibratedLeftImage, grayLeft, CV_BGR2GRAY);
		cv::Mat smallLeftImage(cv::saturate_cast<int>(calibratedLeftImage.rows/SCALE), cv::saturate_cast<int>(calibratedLeftImage.cols/SCALE), CV_8UC1);
		cv::cvtColor(calibratedRightImage, grayRight, CV_BGR2GRAY);
		cv::Mat smallRightImage(cv::saturate_cast<int>(calibratedRightImage.rows/SCALE), cv::saturate_cast<int>(calibratedRightImage.cols/SCALE), CV_8UC1);

		// 処理時間短縮のために画像を縮小
		cv::resize(grayLeft, smallLeftImage, smallLeftImage.size(), 0, 0, cv::INTER_LINEAR);
		cv::equalizeHist( smallLeftImage, smallLeftImage);
		cv::resize(grayRight, smallRightImage, smallRightImage.size(), 0, 0, cv::INTER_LINEAR);
		cv::equalizeHist( smallRightImage, smallRightImage);

		std::vector<cv::Rect> faceLeft; // マルチスケール（顔）探索xo
		std::vector<cv::Rect> faceRight;
		// 画像，出力矩形，縮小スケール，最低矩形数，（フラグ），最小矩形
		cascade.detectMultiScale(smallLeftImage, faceLeft, 1.1, 2, CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
		cascade.detectMultiScale(smallRightImage, faceRight, 1.1, 2, CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));

		// 結果の描画
		std::vector<cv::Rect>::const_iterator l = faceLeft.begin();
		cv::Point leftCenter;
		for(; l != faceLeft.end(); ++l)
		{
			//cv::Point leftCenter;
			int leftRadius;
			leftCenter.x = cv::saturate_cast<int>((l->x + l->width*0.5)*SCALE);
			leftCenter.y = cv::saturate_cast<int>((l->y + l->height*0.5)*SCALE);
			leftRadius = cv::saturate_cast<int>((l->width + l->height)*0.25*SCALE);
			cv::circle( calibratedLeftImage, leftCenter, leftRadius, cv::Scalar(80,80,255), 3, 8, 0 );
		}
		std::vector<cv::Rect>::const_iterator r = faceRight.begin();
		cv::Point rightCenter;
		for(; r != faceRight.end(); ++r)
		{
			//cv::Point rightCenter;
			int rightRadius;
			rightCenter.x = cv::saturate_cast<int>((r->x + r->width*0.5)*SCALE);
			rightCenter.y = cv::saturate_cast<int>((r->y + r->height*0.5)*SCALE);
			rightRadius = cv::saturate_cast<int>((r->width + r->height)*0.25*SCALE);
			cv::circle( calibratedRightImage, rightCenter, rightRadius, cv::Scalar(80,80,255), 3, 8, 0 );
		}

		distance = ((CAMERA / ((double)leftCenter.x - (double)rightCenter.x)) * FOCUS);
		std::cout << "distance = "<< distance << "mm" << std::endl;

		cv::imshow( leftWindow, calibratedLeftImage );
		cv::imshow( rightWindow, calibratedRightImage);
		}
	}