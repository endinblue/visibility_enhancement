#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <string>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <ctime>
#include <math.h>
#include<algorithm>
using namespace cv;
using namespace std;

int kernel_size = 5;
int clahe_time = 0;
int degrade_time = 0;
int local_time = 0;
int filter_time = 0;

//이미지의 grayscale value를 luminance value로 계산

//***************************************************************************************
//인수로 이미지정보 (Mat 형)와 C_r, L_w 정보를 넘겨주어야 함
//***************************************************************************************

Mat grayscale_to_lum(Mat image, double C_r, double L_w) {
	double gamma = 2.2;
	double mul = (L_w - (L_w / C_r));
	image.convertTo(image, CV_32FC1, 1.f/gamma);
	cv::pow(image,2.2,image);
	image = mul * image + L_w / C_r;
	return image;
}

//이미지의 최대, 최솟값 thresholding
void Clamp(std::vector<double>& input_image, double min_value, double max_value) // removed height and width
{
	for (double& d : input_image)
	{
		d = std::max(d, min_value);
	}
	for (double& d : input_image)
	{
		d = std::min(d, max_value);
	}
}

// Saturation (S) 값에 대한 piece-wise 변형을 위한 Look-up table 계산
// global network에서 나온 구간별 기울기 값을 float형 배열에 담아 넘겨줌
// 기울기가 7개일때를 가정하여 구성된 코드이므로 기울기 갯수가 변경될 시 수정 필요

vector<double> SmakeLUT(float* S) {
	vector<double> Stable;
	for (int i = 0; i < 32; i++) {	
		Stable.push_back(*S * i);
	}
	for (int i = 32; i < 64; i++) {
		Stable.push_back(*(S+1) * (i-32)+ Stable[31]+1);
	}
	for (int i = 64; i < 96; i++) {
		Stable.push_back(*(S + 2) * (i - 64) + Stable[63]+1);
	}
	for (int i = 96; i < 128; i++) {
		Stable.push_back(*(S + 3) * (i - 96) + Stable[95] + 1);
	}
	for (int i = 128; i < 160; i++) {
		Stable.push_back(*(S + 4) * (i - 128) + Stable[127] + 1);
	}
	for (int i = 160; i < 192; i++) {
		Stable.push_back(*(S + 5) * (i - 160) + Stable[159] + 1);
	}
	for (int i = 192; i < 224; i++) {
		Stable.push_back(*(S + 6) * (i - 192) + Stable[191] + 1);
	}
	for (int i = 224; i < 256; i++) {
		Stable.push_back(((255-Stable[223]+1)/(225-224))*(i-224)+Stable[223]+1);
	}
	Clamp(Stable, 0, 255);
	return Stable;
}

// V 값에 대한 piece-wise 변형을 위한 Look-up table 계산
// global network에서 나온 구간별 기울기 값을 float형 배열에 담아 넘겨줌
// 기울기가 7개일때를 가정하여 구성된 코드이므로 기울기 갯수가 변경될 시 수정 필요

vector<double> VmakeLUT(float* S) {
	vector<double> Stable;
	for (int i = 0; i < 32; i++) {
		Stable.push_back(*S * i);
	}
	for (int i = 32; i < 64; i++) {
		Stable.push_back(*(S + 1) * (i - 32) + Stable[31] + 1);
	}
	for (int i = 64; i < 96; i++) {
		Stable.push_back(*(S + 2) * (i - 64) + Stable[63] + 1);
	}
	for (int i = 96; i < 128; i++) {
		Stable.push_back(*(S + 3) * (i - 96) + Stable[95] + 1);
	}
	for (int i = 128; i < 160; i++) {
		Stable.push_back(*(S + 4) * (i - 128) + Stable[127] + 1);
	}
	for (int i = 160; i < 192; i++) {
		Stable.push_back(*(S + 5) * (i - 160) + Stable[159] + 1);
	}
	for (int i = 192; i < 224; i++) {
		Stable.push_back(*(S + 6) * (i - 192) + Stable[191] + 1);
	}
	for (int i = 224; i < 256; i++) {
		Stable.push_back(((255 - Stable[223] + 1) / (225 - 224)) * (i - 224) + Stable[223] + 1);
	}
	Clamp(Stable, 0, 255);
	return Stable;
}

bool sortbysec(const pair<float, int>& a, const pair<float, int>& b)
{
	return (a.first < b.first);
}
int main() {

	//image read
	string inputpath = "C:/Users/SEC/Desktop/test/source_image_crop_480";
	Mat bgr = imread(inputpath+".png");

	
	//image size
	cv::Size imageSize = bgr.size();
	int height = imageSize.height;
	int width = imageSize.width;

	string image_class;
	Scalar sourceMean = cv::mean(bgr);
	float *slopes_S;
	float *slopes_V;
	
	/********************Stage 1 : 밝기에 따른 calss 분류 **************************/
	//image 평균 밝기에 따른 class 분류. 현재 기준값은 73,187로 이에 따라 dark, bright, medium으로 구분
	if (sourceMean[0] <= 73)
		image_class = "dark";
	else if (sourceMean[0] >= 187)
		image_class = "bright";
	else
		image_class = "medium";

	//밝기별 class에 따른 S,V의 기울기. S와 V의 기울기가 update 되었다면 수정 필요
	if (image_class == "bright") {
		float slopS[7] = { 1.2535, 1.0885, 1.1234, 1.0430, 1.7176, 0.7589, 0.7873 };
		slopes_S = slopS;
		float slopV[7] = { 2.4805, 1.3927, 1.0929, 1.0794, 0.6619, 0.3859, 0.4994 };
		slopes_V = slopV;
	}
	else if (image_class == "medium") {
		float slopS[7] = { 1.0477, 0.9283, 0.9558, 0.8962, 0.9111, 1.1212, 0.4209 };
		float slopV[7] = { 2.5983, 1.5671, 1.5049, 1.1290, 0.5783, 0.2880, 0.3227 };
		slopes_S = slopS;
		slopes_V = slopV;
	}
	else{
		float slopS[7] = { 1.0922, 1.0296, 1.0504, 1.1474, 1.1533, 1.2540, 1.0564 };
		float slopV[7] = { 3.1607, 1.9947, 1.6872, 0.8078, 0.0000, 0.4100, 0.4767 };
		slopes_S = slopS;
		slopes_V = slopV;
	}

	// 연산에 필요한 변수들 선언
	Mat imgHSV;
	Mat HSVchannel[3];
	Mat mergedHSV;
	Mat outputGamma;
	Mat lab;
	Mat Labchannel[3];
	Mat BGRchannel[3];
	Mat Resultchannel[3];
	Mat Originchannel[3];
	Mat CLAHE;
	Mat dst;
	Mat outputCLAHE;
	Mat origin_result_gray;
	Mat output_result_gray;
	Mat degoutput = Mat(imageSize, CV_32FC3);
	Mat totaloutput = Mat(imageSize, CV_8UC3);
	Mat gamma_gray = Mat(imageSize, CV_32FC1);
	Mat deg_gray = Mat(imageSize, CV_32FC1);
	Mat img_average = Mat(imageSize, CV_32FC1);
	Mat source_square = Mat(imageSize, CV_32FC1);
	Mat source_local = Mat(imageSize, CV_32FC1);
	Mat deg_average = Mat(imageSize, CV_32FC1);
	Mat deg_square = Mat(imageSize, CV_32FC1);
	Mat deg_local = Mat(imageSize, CV_32FC1);
	Mat localmap = Mat(imageSize, CV_32FC1);

	/********************Stage 2 : global 연산 수행 **************************/
	//넘겨받은 S와 V의 기울기 값에 대해 look up table 계산
	vector<double> Stable = SmakeLUT(slopes_S);
	vector<double> Vtable = VmakeLUT(slopes_V);

	//입력이미지를 각각 gray scale, hsv 색공간, lab 색공간의로 변형
	cvtColor(bgr, origin_result_gray, cv::COLOR_BGR2GRAY);
	cvtColor(bgr, imgHSV, cv::COLOR_BGR2HSV);
	cvtColor(bgr, lab, cv::COLOR_BGR2Lab);

	//변형된 hsv와 lab 색공간 이미지의 각 채널을 배열에 각각 분리하여 저장
	split(imgHSV,HSVchannel);
	split(lab, Labchannel);

	unsigned char* pDataS = HSVchannel[1].data;
	unsigned char* pDataV = HSVchannel[2].data;

	//lookup table에 따른 이미지 변환. 이때 변환 hsv channel대해 작동함
	for (int i = 0; i < width; ++i)
	{
		for (int j = 0; j < height; ++j)
		{
			pDataS[j * width + i] = Stable[pDataS[j * width + i]];
			pDataV[j * width + i] = Vtable[pDataV[j * width + i]];
		}
	}
	
	/*Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = saturate_cast<float16_t> (pow(i / 255.0, gamma) * 255);
	LUT(image, lookUpTable, image);*/

	memcpy(HSVchannel[1].data, pDataS, sizeof(unsigned char) * width * height);
	memcpy(HSVchannel[2].data, pDataV, sizeof(unsigned char) * width * height);
	merge(HSVchannel, 3, mergedHSV);
	
	//lookup table을 통해 global하게 piecewise 변환된 hsv 공간의 이미지를 rgb공간으로 변환하여 outputGamma에 저장
	cvtColor(mergedHSV, outputGamma, cv::COLOR_HSV2BGR);
	

	// clahe 수행후 outputCLAHE에 저장
	Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0,Size(10,10));	
	clahe->apply(Labchannel[0], dst);
	dst.copyTo(Labchannel[0]);
	merge(Labchannel,3, lab);
	cvtColor(lab, outputCLAHE, cv::COLOR_Lab2BGR);
	
	/*imwrite("clahe.png", outputCLAHE);*/
	
	/********************Stage 3 : degradation 모델링 값 도출 **************************/
	
	// degradation 모델링을 수행하기 위한 변수 지정
	double L_ws = 400; double L_wd = 400;
	double C_rs = 150; double C_rd = 150;
	double L_e = 0.02 * L_ws;
	
	double min_gray, max_gray, C_rs_p, L_wd_p, G_wd_p, L0;;
	double gamma = 2.2; double jisu = 1 / 2.2;
	double factorMul, factorUnder;
	Mat ratio;
	output_result_gray = origin_result_gray.clone();
	
	// lookup table을 통해 global 수행을 마친 RGB공간의 outputGamma를 grayscale로 변경하여 output_result_gray에 저장
	cvtColor(outputGamma, output_result_gray, cv::COLOR_BGR2GRAY);
	
	// 외부조도에 의해 감소되는 인지가능한 콘트라스의 변화 (Reduction of the perceptible contrast)
	// 논문 값 인용
	C_rs_p = (L_ws + L_e) / ((L_ws / C_rs) + L_e);
	L_wd_p = L_wd * (C_rs_p / C_rd);
	G_wd_p = pow((C_rs_p-1)/(C_rd -1) ,jisu) * 256;

	//global 연산이 수행된 이미지(grayscale)를 lumininace 값으로 변환
	output_result_gray = grayscale_to_lum(output_result_gray, C_rs, L_ws);

	//외부조도에 의해 감소되는 인지가능한 grayscale 차이. (Reduction of the noticeable grayscale differences)
	cv::minMaxLoc(output_result_gray, &min_gray, &max_gray);
	L0 = max_gray + L_e;
	output_result_gray = (output_result_gray + L_e) / L0;
	output_result_gray = output_result_gray * L_wd;

	factorMul = L_wd / C_rd;
	factorUnder = L_wd - factorMul;
	output_result_gray = (output_result_gray - factorMul)/factorUnder;
	cv::pow(output_result_gray, jisu, output_result_gray);
	output_result_gray = output_result_gray * 256;
	
	cv::minMaxLoc(output_result_gray, &min_gray, &max_gray);
	double num = 1 / (max_gray - min_gray)*G_wd_p;
	output_result_gray = output_result_gray-min_gray;
	output_result_gray = output_result_gray * num;
	
    origin_result_gray.convertTo(origin_result_gray, CV_32FC1);
	bgr.convertTo(bgr, CV_32FC1);
	ratio = origin_result_gray.clone();
	cv::divide(output_result_gray, (origin_result_gray + 1e-7),ratio);

	/*cout << max_gray.at<float>(1, 1) << endl;*/
	//외부조도에 의해 감소되는 시인성을 모델링한 factor 값을 각 채널에 대해 곱해, 최종적인 모델링 이미지를 도출
	
	split(bgr, BGRchannel);
	cv::multiply(ratio, BGRchannel[0],BGRchannel[0]);
	cv::multiply(ratio, BGRchannel[1], BGRchannel[1]);
	cv::multiply(ratio, BGRchannel[2], BGRchannel[2]);
	
	BGRchannel[0].convertTo(BGRchannel[0], CV_8UC1);
	BGRchannel[1].convertTo(BGRchannel[1], CV_8UC1);
	BGRchannel[2].convertTo(BGRchannel[2], CV_8UC1);
	merge(BGRchannel, 3, degoutput);
	outputGamma.convertTo(outputGamma, CV_32FC1);
	cvtColor(degoutput, deg_gray, cv::COLOR_BGR2GRAY);
	deg_gray.convertTo(deg_gray, CV_32FC1);
	cvtColor(outputGamma, gamma_gray, cv::COLOR_BGR2GRAY);


	/********************Stage 4: Local 연산 수행 **************************/
	//local map 구하기
	Scalar source_gray_mean = mean(origin_result_gray); 
	
	// mask 가 (5,5) 인 평균필터 커널
	Mat avg_kernel = Mat::ones(5, 5, CV_32F) / 25;
	
	//img_average = (source_image(grayscale)의 5*5kernel 평균)^2
	filter2D(origin_result_gray, img_average, -1, avg_kernel, Point(-1, -1), (0, 0), BORDER_REFLECT);
	cv::pow(img_average, 2, img_average);
	cv::pow(origin_result_gray, 2, source_square);
	
	//source_square = (source_image(grayscale))^2의 5*5kernel 평균
	filter2D(source_square, source_square, -1, avg_kernel, Point(-1, -1), (0, 0), BORDER_REFLECT);
	
	//source_local = source_image의 local contrast (E[I]^2 - E[I^2])
	cv::subtract(source_square, img_average, source_local);

	//같은방식으로 degradation modeling된 이미지에 대해서도 local contrast 계산
	filter2D(deg_gray, deg_average, -1, avg_kernel, Point(-1, -1), (0, 0), BORDER_REFLECT);
	cv::pow(deg_average, 2, deg_average);
	cv::pow(deg_gray, 2, deg_square);
	filter2D(deg_square, deg_square, -1, avg_kernel, Point(-1, -1), (0, 0), BORDER_REFLECT);
	cv::subtract(deg_square, deg_average, deg_local);

	//최종적인 local map= source_local/deg_local
	//local map 값이 클수록 degradation modeling된 이미지가 original image보다 시인성이 크게 저하됨
	cv::divide(source_local, (deg_local + 1e-7), localmap);
	localmap = cv::abs(localmap);
	int allpixel = bgr.rows * bgr.cols * bgr.channels();
	
	cv::Size localimageSize = localmap.size();
	Mat new_local_map = Mat(localimageSize, CV_32FC1);
	Mat bright_new_local_map = Mat(localimageSize, CV_32FC1);
	Mat norm_local_map = Mat(localimageSize, CV_8UC1);
	Mat local_map_percentile = Mat(imageSize, CV_32FC1);
	std::vector<float> sortarray(localmap.rows* localmap.cols);
	std::vector<float> originarray(localmap.rows* localmap.cols);
	std::vector<float> localmaparray(localmap.rows* localmap.cols);
	std::vector<float> originlocalmaparray(localmap.rows* localmap.cols);
	//std::vector<int> rankdata_max_array(localmap.rows* localmap.cols);
	float* rankdata_max_array = new float[localmap.rows * localmap.cols];
	std::vector<pair<float,int>> pairlocalmaparray(localmap.rows* localmap.cols);

	if (origin_result_gray.isContinuous()) {
		sortarray.assign((float*)origin_result_gray.data, (float*)origin_result_gray.data + origin_result_gray.total());
		originarray.assign((float*)origin_result_gray.data, (float*)origin_result_gray.data + origin_result_gray.total());
	}
	if (localmap.isContinuous()) {
		localmaparray.assign((float*)localmap.data, (float*)localmap.data + localmap.total());
		originlocalmaparray.assign((float*)localmap.data, (float*)localmap.data + localmap.total());		
	}
	

	if (image_class == "bright") {

		sort(sortarray.begin(), sortarray.end());
		// 20% percentile 값 찾기
		float maxstandard = sortarray[int(sortarray.size() * 2 / 10)];
		if (maxstandard > source_gray_mean[0])
			maxstandard = source_gray_mean[0];
		for (int i = 0; i < originarray.size(); i++) {
			if (originarray[i] > maxstandard) {
				localmaparray[i] = 0;
			}
			
		}
		//localmap rankdata(max) 정렬하기
		for (int i = 0; i < localmap.rows * localmap.cols; i++) {
			pairlocalmaparray[i] = make_pair(localmaparray[i], i);			
		}
		
		//rankdata 정렬을 위해 pair형으로 value와 index를 새로운 vector에 넣어줌
		//rankdata는 value 값이 중요한것이 아니라 값에 대한 순서를 매기는 것이므로 index정보가 포함되어야함
		sort(pairlocalmaparray.begin(), pairlocalmaparray.end(), sortbysec);
	
		int samecount = 0;
		double totalpixelrank = localmap.rows * localmap.cols;
		float rankvalue = localmap.rows * localmap.cols;
		float previousvalue = pairlocalmaparray[totalpixelrank - 1].first;
		float percentnum = 100.00;
		for (int i = totalpixelrank-1; i >=0; i--) {
			float currentvalue = pairlocalmaparray[i].first;
			int setindex = pairlocalmaparray[i].second;
			
			if (currentvalue == previousvalue) {
				samecount++;
				rankdata_max_array[setindex] = int(float(rankvalue*100)/totalpixelrank);
				rankdata_max_array[setindex] = rankdata_max_array[setindex] / percentnum;
			
			}
			else {
				rankvalue -= samecount;
				rankdata_max_array[setindex] = int(float(rankvalue*100)/totalpixelrank);
				rankdata_max_array[setindex] = rankdata_max_array[setindex] / percentnum;
				samecount = 1;
				previousvalue = currentvalue;
			}
		}
		
		memcpy(new_local_map.data, rankdata_max_array, sizeof(CV_32FC1) * width * height);
		cv::multiply(new_local_map, (1-(origin_result_gray/255.00)) , local_map_percentile);
		
		cv::minMaxLoc(new_local_map, &min_gray, &max_gray);
		new_local_map = new_local_map - float(min_gray);
		new_local_map = new_local_map / float(max_gray - min_gray);

		split(outputGamma, Resultchannel);
		split(bgr, Originchannel);
		for (int i = 0; i < 3; i++) {
			cv::multiply(Resultchannel[i], (1-new_local_map),Resultchannel[i]);
			cv::multiply(Originchannel[i], new_local_map, Originchannel[i]);
			cv::add(Resultchannel[i], Originchannel[i], Resultchannel[i]);		
			Resultchannel[i].convertTo(Resultchannel[i], CV_8UC1);
		}
		merge(Resultchannel, 3, totaloutput);

		cv::imwrite(inputpath + "_result.png", totaloutput);
		
	}

	else if (image_class == "dark") {

		sort(sortarray.begin(), sortarray.end());
		// 80% percentile 값 찾기
		float maxstandard = sortarray[int(sortarray.size() * 8 / 10)];
		if (maxstandard < source_gray_mean[0])
			maxstandard = source_gray_mean[0];
		for (int i = 0; i < originarray.size(); i++) {
			if (originarray[i] < maxstandard) {
				localmaparray[i] = 0;
			}
		}
		//localmap rankdata(max) 정렬하기
		for (int i = 0; i < localmap.rows * localmap.cols; i++) {
			pairlocalmaparray[i] = make_pair(localmaparray[i], i);
		}


		sort(pairlocalmaparray.begin(), pairlocalmaparray.end(), sortbysec);

		int samecount = 0;
		double totalpixelrank = localmap.rows * localmap.cols;
		float rankvalue = localmap.rows * localmap.cols;
		float previousvalue = pairlocalmaparray[totalpixelrank - 1].first;
		float percentnum = 100.00;
		for (int i = totalpixelrank - 1; i >= 0; i--) {
			float currentvalue = pairlocalmaparray[i].first;
			int setindex = pairlocalmaparray[i].second;

			if (currentvalue == previousvalue) {
				samecount++;
				rankdata_max_array[setindex] = int(float(rankvalue * 100) / totalpixelrank);
				rankdata_max_array[setindex] = rankdata_max_array[setindex] / percentnum;

			}
			else {
				rankvalue -= samecount;
				rankdata_max_array[setindex] = int(float(rankvalue * 100) / totalpixelrank);
				rankdata_max_array[setindex] = rankdata_max_array[setindex] / percentnum;
				samecount = 1;
				previousvalue = currentvalue;
			}
		}

		memcpy(new_local_map.data, rankdata_max_array, sizeof(CV_32FC1) * width * height);
		cv::multiply(new_local_map, (1 - (origin_result_gray / 255.00)), local_map_percentile);

		cv::minMaxLoc(new_local_map, &min_gray, &max_gray);
		new_local_map = new_local_map - float(min_gray);
		new_local_map = new_local_map / float(max_gray - min_gray);

		split(outputGamma, Resultchannel);
		split(bgr, Originchannel);
		for (int i = 0; i < 3; i++) {
			cv::multiply(Resultchannel[i], (1 - new_local_map), Resultchannel[i]);
			cv::multiply(Originchannel[i], new_local_map, Originchannel[i]);
			cv::add(Resultchannel[i], Originchannel[i], Resultchannel[i]);
			Resultchannel[i].convertTo(Resultchannel[i], CV_8UC1);
		}
		merge(Resultchannel, 3, totaloutput);

		cv::imwrite(inputpath + "_result.png", totaloutput);

	}

	else {

	sort(sortarray.begin(), sortarray.end());
	// 80% percentile 값 찾기
	std::vector<float> darklocalmaparray(localmap.rows* localmap.cols);
	std::vector<float> brightlocalmaparray(localmap.rows* localmap.cols);

	if (localmap.isContinuous()) {
		darklocalmaparray.assign((float*)localmap.data, (float*)localmap.data + localmap.total());
		//originlocalmaparray.assign((float*)localmap.data, (float*)localmap.data + localmap.total());
	}
	if (localmap.isContinuous()) {
		brightlocalmaparray.assign((float*)localmap.data, (float*)localmap.data + localmap.total());
		//originlocalmaparray.assign((float*)localmap.data, (float*)localmap.data + localmap.total());
	}

	float maxstandard = sortarray[int(sortarray.size() * 8 / 10)];
	if (maxstandard < source_gray_mean[0])
		maxstandard = source_gray_mean[0];

	for (int i = 0; i < originarray.size(); i++) {
		if (originarray[i] < maxstandard) {
			darklocalmaparray[i] = 0;
		}
	}
	//localmap rankdata(max) 정렬하기
	for (int i = 0; i < localmap.rows * localmap.cols; i++) {
		pairlocalmaparray[i] = make_pair(darklocalmaparray[i], i);
	}
	sort(pairlocalmaparray.begin(), pairlocalmaparray.end(), sortbysec);

	int samecount = 0;
	double totalpixelrank = localmap.rows * localmap.cols;
	float rankvalue = localmap.rows * localmap.cols;
	float previousvalue = pairlocalmaparray[totalpixelrank - 1].first;
	float percentnum = 100.00;
	for (int i = totalpixelrank - 1; i >= 0; i--) {
		float currentvalue = pairlocalmaparray[i].first;
		int setindex = pairlocalmaparray[i].second;

		if (currentvalue == previousvalue) {
			samecount++;
			rankdata_max_array[setindex] = int(float(rankvalue * 100) / totalpixelrank);
			//rankdata_max_array[setindex] = rankdata_max_array[setindex] / percentnum;

		}
		else {
			rankvalue -= samecount;
			rankdata_max_array[setindex] = int(float(rankvalue * 100) / totalpixelrank);
			//rankdata_max_array[setindex] = rankdata_max_array[setindex] / percentnum;
			samecount = 1;
			previousvalue = currentvalue;
		}
	}

	float darkbignum = 0;
	float darkbigsum = 0;
	for (int i = 0; i < totalpixelrank - 1; i++) {
		rankdata_max_array[i] = rankdata_max_array[i] * originarray[i] / 255.00;
		if (rankdata_max_array[i] >= maxstandard) {
			darkbignum+=1;
		}
		rankdata_max_array[i] = rankdata_max_array[i] / percentnum;
	}

	memcpy(new_local_map.data, rankdata_max_array, sizeof(CV_32FC1)* width* height);
	if (darkbignum > 0.6*totalpixelrank)
		new_local_map = new_local_map * 0.5;


	float minstandard = sortarray[int(sortarray.size() * 2 / 10)];
	if (minstandard > source_gray_mean[0])
		minstandard = source_gray_mean[0];

	for (int i = 0; i < originarray.size(); i++) {
		if (originarray[i] > minstandard) {
			brightlocalmaparray[i] = 0;
		}
	}

	for (int i = 0; i < localmap.rows * localmap.cols; i++) {
		pairlocalmaparray[i] = make_pair(brightlocalmaparray[i], i);
	}
	sort(pairlocalmaparray.begin(), pairlocalmaparray.end(), sortbysec);

	rankvalue = 1;
	previousvalue = pairlocalmaparray[0].first;
	percentnum = 100.00;

	for (int i = 0; i < totalpixelrank ; i++) {
		float currentvalue = pairlocalmaparray[i].first;
		int setindex = pairlocalmaparray[i].second;

		if (currentvalue == previousvalue) {
			rankdata_max_array[setindex] = rankvalue;
			//rankdata_max_array[setindex] = rankdata_max_array[setx] / percentnum;
		}
		else {
			rankvalue += 1;
			rankdata_max_array[setindex] = rankvalue;
			//rankdata_max_array[setindex] = rankdata_max_array[setindex] / percentnum;
			previousvalue = currentvalue;
		}
	}
	for (int i = 0; i < totalpixelrank; i++) {
		rankdata_max_array[i] = int(rankdata_max_array[i] / (rankvalue / float(100.00)));
		rankdata_max_array[i] = rankdata_max_array[i] *(float(1.00)- originarray[i]/float(255.00));
		if (rankdata_max_array[i] > minstandard)
			rankdata_max_array[i] = 0;
		rankdata_max_array[i] = rankdata_max_array[i] / percentnum;
	}
	memcpy(bright_new_local_map.data, rankdata_max_array, sizeof(CV_32FC1)* width* height);
		
	cv::add(new_local_map,bright_new_local_map, new_local_map);
	
	/*#dark_new_local_map[source_gray < maxstandard] = 0
		if np.mean(dark_new_local_map[source_gray >= maxstandard]) > 0.6:
	dark_new_local_map = dark_new_local_map * 0.5*/

	cv::minMaxLoc(new_local_map, &min_gray, &max_gray);
	if (max_gray < 1)
		max_gray = 1;
	new_local_map = new_local_map - float(min_gray);
	new_local_map = new_local_map / float(max_gray - min_gray);

	split(outputGamma, Resultchannel);
	split(bgr, Originchannel);
	for (int i = 0; i < 3; i++) {
		cv::multiply(Resultchannel[i], (1 - new_local_map), Resultchannel[i]);
		cv::multiply(Originchannel[i], new_local_map, Originchannel[i]);
		cv::add(Resultchannel[i], Originchannel[i], Resultchannel[i]);
		Resultchannel[i].convertTo(Resultchannel[i], CV_8UC1);
	}
	merge(Resultchannel, 3, totaloutput);
	cv::imwrite(inputpath+"_result.png", totaloutput);

	}
	
	
	
}