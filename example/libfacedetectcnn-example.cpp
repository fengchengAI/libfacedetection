#include <cstdio>
#include <opencv2/opencv.hpp>
#include "facedetectcnn.h"

//define the buffer size. Do not change the size!
#define DETECT_BUFFER_SIZE 0x20000
using namespace cv;
using namespace std;

int main()
{

	//load an image and convert it to gray (single-channel)
	Mat image = imread("/home/feng/1.jpg");
	if(image.empty())
	{
		//fprintf(stderr, "Can not load the image file %s.\n", argv[1]);
		std::cout<<"empty";
		return -1;
	}

	int * pResults = NULL; 
    //pBuffer is used in the detection functions.
    //If you call functions in multiple threads, please create one buffer for each thread!
    unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);


    if(!pBuffer)
    {
        fprintf(stderr, "Can not alloc buffer.\n");
        return -1;
    }

	///////////////////////////////////////////
	// CNN face detection 
	// Best detection rate
	//////////////////////////////////////////
	//!!! The input image must be a BGR one (three-channel) instead of RGB
	//!!! DO NOT RELEASE pResults !!!

    pResults = facedetect_cnn(pBuffer, (unsigned char*)(image.ptr(0)), image.cols, image.rows, (int)image.step);
/*rows = 2318
 *cols = 1084
 *step = 1084*3
 */
    printf("%d faces detected.\n", (pResults ? *pResults : 0));
	Mat result_cnn = image.clone();
	//print the detection results
	for(int i = 0; i < (pResults ? *pResults : 0); i++)  //pResults 的首地址存的是face的个数
	{
        short * p = ((short*)(pResults+1))+5*i;
		int x = p[0];
		int y = p[1];
		int w = p[2];
		int h = p[3];
		int confidence = p[4];
	//	int angle = p[5];

		printf("face_rect=[%d, %d, %d, %d], confidence=%d\n", x,y,w,h,confidence);
		rectangle(result_cnn, Rect(x, y, w, h), Scalar(0, 255, 0), 2);
	}
	imshow("result_cnn", result_cnn);

	waitKey();

    //release the buffer
    free(pBuffer);

	return 0;
}
