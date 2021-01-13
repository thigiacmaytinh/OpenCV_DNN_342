#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/trace.hpp>


#include <fstream>
#include <iostream>
#include <cstdlib>
#include <time.h>

//TGMT: thigiacmaytinh.com
#include "TGMTConfig.h"
#include "TGMTdebugger.h"
#include "stdafx.h"

using namespace cv::dnn;

std::string CLASSES[] = { "background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor" };


int main(int argc, char **argv)
{
	if (!GetTGMTConfig()->LoadSettingFromFile())
	{
		PrintError("Can not read config");
		WriteLog("Can not read config");
		return 0;
	}

	CV_TRACE_FUNCTION();
	std::string modelTxt = "MobileNetSSD_deploy.prototxt";
	std::string modelBin = "MobileNetSSD_deploy.caffemodel";

	std::string imageFile = GetTGMTConfig()->ReadValueString("ObjectDetection_MobileNet_SSD", "input");

	time_t timeLoadModel = clock();
	Net net = cv::dnn::readNetFromCaffe(modelTxt, modelBin);
	std::cout << "Load model elapsed: " << clock() - timeLoadModel << "ms\n";

	if (net.empty())
	{
		std::cerr << "Can't load network by using the following files: " << std::endl;
		std::cerr << "prototxt:   " << modelTxt << std::endl;
		std::cerr << "caffemodel: " << modelBin << std::endl;
		exit(-1);
	}
	cv::Mat img = cv::imread(imageFile);
	if (img.empty())
	{
		std::cerr << "Can't read image from the file: " << imageFile << std::endl;
		exit(-1);
	}

	cv::Mat img2;
	cv::resize(img, img2, cv::Size(300, 300));
	cv::Mat inputBlob = blobFromImage(img2, 0.007843, cv::Size(300, 300), cv::Scalar(127.5, 127.5, 127.5), false);


	time_t timeProcess = clock();
	net.setInput(inputBlob, "data");
	cv::Mat detection = net.forward("detection_out");
	cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	std::cout << "Process elapsed: " << clock() - timeProcess << "ms\n";


	std::ostringstream ss;
	float confidenceThreshold = 0.2;
	for (int i = 0; i < detectionMat.rows; i++)
	{
		float confidence = detectionMat.at<float>(i, 2);

		if (confidence > confidenceThreshold)
		{
			int idx = static_cast<int>(detectionMat.at<float>(i, 1));
			int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * img.cols);
			int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * img.rows);
			int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * img.cols);
			int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * img.rows);

			cv::Rect object(xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom);

			rectangle(img, object, cv::Scalar(0, 255, 0), 2);

			std::cout << CLASSES[idx] << ": " << confidence << std::endl;

			ss.str("");
			ss << confidence;
			std::string conf(ss.str());
			std::string label = CLASSES[idx] + ": " + conf;
			int baseLine = 0;
			cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
			putText(img, label, cv::Point(xLeftBottom, yLeftBottom),
				cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		}
	}
	cv::imshow("detections", img);
	cv::waitKey();

	return 0;
}