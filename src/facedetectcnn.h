
#pragma once

int * facedetect_cnn(unsigned char * result_buffer, //buffer memory for storing face detection results, !!its size must be 0x20000 Bytes!!
                    unsigned char * rgb_image_data, int width, int height, int step); //input image, it must be BGR (three channels) insteed of RGB image!

#define _MAX_UINT8_VALUE 255

#define _MALLOC_ALIGN 128

#include <cstring>
#include <vector>
#include <iostream>

using namespace std;

void* myAlloc(size_t size);
void myFree_(void* ptr);
#define myFree(ptr) (myFree_(*(ptr)), *(ptr)=0);

#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

typedef struct FaceRect_
{
    float score;
    int x;
    int y;
    int w;
    int h;
}FaceRect;
    
template <class T>
class CDataBlob
{
public:
    T * data;
	int width;
	int height;
	int channels;
    int channelStep;
    float scale;  //这是什么
public:
	CDataBlob() {
        data = 0;
		width = 0;
		height = 0;
        channels = 0;
        channelStep = 0;
        scale = 1.0f;
	}
	CDataBlob(int w, int h, int c)
	{
        data = 0;
        create(w, h, c);
	}
	~CDataBlob()
	{
        setNULL();
	}

    void setNULL()
    {
        if (data)
            myFree(&data);
        width = height = channels = channelStep = 0;
        scale = 1.0f;
    }
	bool create(int w, int h, int c)
	{
        setNULL();

		width = w;
		height = h;
        channels = c;

        //alloc space for int8 array
        int remBytes = (sizeof(T)* channels) % (_MALLOC_ALIGN / 8);
        if (remBytes == 0)
            channelStep = channels * sizeof(T);
        else
            channelStep = (channels * sizeof(T)) + (_MALLOC_ALIGN / 8) - remBytes;
        data = (T*)myAlloc(width * height * channelStep);

        if (data == NULL)
        {
            cerr << "Cannot alloc memeory for uint8 data blob: "
                << width << "*"
                << height << "*"
                << channels << endl;
            return false;
        }

        memset(data, 0, width * height * channelStep);

        //the following code is faster than memset
        //but not only the padding bytes are set to zero.
        //BE CAREFUL!!!
        // 因为这里进行了通道对齐和指针对齐，所以对与没有用到的会进行初始化０
        /*
        for (int r = 0; r < height; r++)
        {
            for (int c = 0; c < width; c++)
            {
                int pixel_end = channelStep / sizeof(T);
                T * pI = (data + (r * width + c) * channelStep /sizeof(T));
                for (int ch = channels; ch < pixel_end; ch++)
                    pI[ch] = 0;
            }
        }*/
        
        return true;
	}

    bool setInt8DataFromCaffeFormat(signed char * pData, int dataWidth, int dataHeight, int dataChannels)
    {
        if (pData == NULL)
        {
            cerr << "The input image data is null." << endl;
            return false;
        }

        if (typeid(signed char) != typeid(T))
        {
            cerr << "Data must be signed char, the same with the source data." << endl;
            return false;
        }

        if (dataWidth != width || dataHeight != height || dataChannels != channels)
        {
            cerr << "The dim of the data can not match that of the Blob." << endl;
            return false;
        }

        for(int row = 0; row < height; row++)
            for (int col = 0; col < width; col++)
            {
                T * p = (data + (width * row + col) * channelStep /sizeof(T));
                for (int ch = 0; ch < channels; ch++)
                {
                    p[ch] = pData[ch * height * width + row * width + col];
                }
            }
        return true;
    }

    bool setDataFrom3x3S2P1to1x1S1P0FromImage(const unsigned char * imgData, int imgWidth, int imgHeight, int imgChannels, int imgWidthStep)
    {//只是在读取image是用到了
        if (imgData == NULL)
        {
            cerr << "The input image data is null." << endl;
            return false;
        }
        if (typeid(unsigned char) != typeid(T))
        {
            cerr << "Data must be unsigned char, the same with the source data." << endl;
            return false;
        }
        if (imgChannels != 3)
        {
            cerr << "The input image must be a 3-channel RGB image." << endl;
            return false;
        }
        //CDataBlob<unsigned char> inputImag;
        //inputImage.setDataFrom3x3S2P1to1x1S1P0FromImage(rgbImageData, width, height, 3, step);
        //rgbImageData是image头地址

        create((imgWidth+1)/2, (imgHeight+1)/2, 27);  //为什么???
        //since the pixel assignment cannot fill all the elements in the blob. 
        //some elements in the blob should be initialized to 0
        memset(data, 0, width * height * channelStep);

        for (int r = 0; r < height; r++)
        {
            for (int c = 0; c < width; c++)
            {
                T * pData = (unsigned char*)data + (r * width + c) * channelStep;
                for (int fy = -1; fy <= 1; fy++)
                {
                    int srcy = r * 2 + fy; //这里可以解释为什么create会有除以２
                    
                    if (srcy < 0 || srcy >= imgHeight) //out of the range of the image
                        continue;

                    for (int fx = -1; fx <= 1; fx++)
                    {
                        int srcx = c * 2 + fx;

                        if (srcx < 0 || srcx >= imgWidth) //out of the range of the image
                            continue;
//     offset = (row * numCols * numChannels) + (col * numChannels) + (channel)
                        const unsigned char * pImgData = imgData + imgWidthStep * srcy + imgChannels * srcx;

                        int output_channel_offset = ((fy
                                + 1) * 3 + fx + 1) * 3; //3x3 filters, 3-channel image

                        pData[output_channel_offset] = (pImgData[0]);
                        pData[output_channel_offset+1] = (pImgData[1]);
                        pData[output_channel_offset+2] = (pImgData[2]);

                    }
                }
            }
        }
        return true;
    }

    T getElement(int x, int y, int channel)
    {
        if (data)
        {
            if (x >= 0 && x < width &&
                y >= 0 && y < height &&
                channel >= 0 && channel < channels)
            {
                T * p = data + (y*width + x)*channelStep/sizeof(T);
                return (p[channel]);
            }
        }
        
        return (T)(0);
    }
};

class Filters {
public:
	vector<CDataBlob<signed char> *> filters;
	int pad;
	int stride;
    float scale; //element * scale = original value
};

bool convertInt2Float(CDataBlob<int> * inputData, CDataBlob<float> * outputData);

bool convolution(CDataBlob<unsigned char> *inputData, const Filters* filters, CDataBlob<int> *outputData);

bool convolution_relu(CDataBlob<unsigned char> *inputData, const Filters* filters, CDataBlob<unsigned char> *outputData);

bool maxpooling2x2S2(const CDataBlob<unsigned char> *inputData, CDataBlob<unsigned char> *outputData);

bool normalize(CDataBlob<unsigned char> * inputOutputData, float * pScale);

bool priorbox(const CDataBlob<unsigned char> * featureData, const CDataBlob<unsigned char> * imageData, int num_sizes, float * pWinSizes, CDataBlob<float> * outputData);

template<typename T>
bool concat4(const CDataBlob<T> *inputData1, const CDataBlob<T> *inputData2, const CDataBlob<T> *inputData3, const CDataBlob<T> *inputData4, CDataBlob<T> *outputData);

/* the input data for softmax must be a vector, the data stored in a multi-channel blob with size 1x1 */
template<typename T>
bool blob2vector(const CDataBlob<T> * inputData, CDataBlob<T> * outputData);

bool softmax1vector2class(CDataBlob<float> *inputOutputData);

bool detection_output(const CDataBlob<float> * priorbox, const CDataBlob<float> * loc, const CDataBlob<float> * conf, float overlap_threshold, float confidence_threshold, int top_k, int keep_top_k, CDataBlob<float> * outputData);

vector<FaceRect> objectdetect_cnn(unsigned char * rgbImageData, int with, int height, int step);
