#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

const char* keys =
    "{ help  h     | | Print help message. }"
    "{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera.}"
    "{ model m     | | Path to a binary file of model contains trained weights. "
                      "It could be a file with extensions .caffemodel (Caffe), "
                      ".pb (TensorFlow), .t7 or .net (Torch), .weights (Darknet) }"
    "{ config c    | | Path to a text file of model contains network configuration. "
                      "It could be a file with extensions .prototxt (Caffe), .pbtxt (TensorFlow), .cfg (Darknet) }"
    "{ framework f | | Optional name of an origin framework of the model. Detect it automatically if it does not set. }"
    "{ classes     | | Optional path to a text file with names of classes. }"
    "{ mean        | | Preprocess input image by subtracting mean values. Mean values should be in BGR order and delimited by spaces. }"
    "{ scale       | 1 | Preprocess input image by multiplying on a scale factor. }"
    "{ width       |   | Preprocess input image by resizing to a specific width. }"
    "{ height      |   | Preprocess input image by resizing to a specific height. }"
    "{ rgb         |   | Indicate that model works with RGB input images instead BGR ones. }"
    "{ backend     | 0 | Choose one of computation backends: "
                        "0: automatically (by default), "
                        "1: Halide language (http://halide-lang.org/), "
                        "2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                        "3: OpenCV implementation }"
    "{ target      | 0 | Choose one of target computation devices: "
                        "0: CPU target (by default), "
                        "1: OpenCL, "
                        "2: OpenCL fp16 (half-float precision), "
                        "3: VPU }";

using namespace cv;
using namespace cv::dnn;

std::vector<std::string> classes;

int main(int argc, char** argv)
{
	cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run classification deep learning networks using OpenCV.");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    float scale = parser.get<float>("scale");
	cv::Scalar mean = parser.get<cv::Scalar>("mean");
    bool swapRB = parser.get<bool>("rgb");
    CV_Assert(parser.has("width"), parser.has("height"));
    int inpWidth = parser.get<int>("width");
    int inpHeight = parser.get<int>("height");
	cv::String model = parser.get<cv::String>("model");
	cv::String config = parser.get<cv::String>("config");
	cv::String framework = parser.get<cv::String>("framework");
    int backendId = parser.get<int>("backend");
    int targetId = parser.get<int>("target");

    // Open file with classes names.
    if (parser.has("classes"))
    {
        std::string file = parser.get<cv::String>("classes");
        std::ifstream ifs(file.c_str());
        if (!ifs.is_open())
            CV_Error(Error::StsError, "File " + file + " not found");
        std::string line;
        while (std::getline(ifs, line))
        {
            classes.push_back(line);
        }
    }

    CV_Assert(parser.has("model"));
    //! [Read and initialize network]
    Net net = readNet(model, config, framework);
    net.setPreferableBackend(backendId);
    net.setPreferableTarget(targetId);
    //! [Read and initialize network]

    //! [Open a video file or an image file or a camera stream]
    cv::VideoCapture cap;

    if (parser.has("input"))
        cap.open(parser.get<cv::String>("input"));
    else
        cap.open(0);
    //! [Open a video file or an image file or a camera stream]

    // Process frames.
	cv::Mat frame, blob;
    while (cv::waitKey(1) < 0)
    {		
		if (cap.isOpened())
			cap >> frame;
		else
			frame = cv::imread(parser.get<cv::String>("input"));
        
        if (frame.empty())
        {
			std::cout << "Cannot load image";
			cv::waitKey();
            break;
        }

        //! [Create a 4D blob from a frame]
        blobFromImage(frame, blob, scale, cv::Size(inpWidth, inpHeight), mean, swapRB, false);
        //! [Create a 4D blob from a frame]

        //! [Set input blob]
        net.setInput(blob);
        //! [Set input blob]
        //! [Make forward pass]
		cv::Mat prob = net.forward();
        //! [Make forward pass]

        //! [Get a class with a highest score]
		cv::Point classIdPoint;
        double confidence;
        minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
        int classId = classIdPoint.x;
        //! [Get a class with a highest score]

        // Put efficiency information.
        std::vector<double> layersTimes;
        double freq = cv::getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        std::string label = cv::format("Inference time: %.2f ms", t);
        putText(frame, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));

        // Print predicted class.
        label = cv::format("%s: %.4f", (classes.empty() ? cv::format("Class #%d", classId).c_str() :
                                                      classes[classId].c_str()),
                                   confidence);
        putText(frame, label, cv::Point(0, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));

        imshow("Deep learning image classification in OpenCV", frame);
		cv::waitKey();
		if (cap.isOpened())
			break;
    }
    return 0;
}
