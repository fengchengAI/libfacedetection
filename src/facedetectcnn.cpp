#include "facedetectcnn.h"
#include <string.h>
#include <cmath>
#include <vector>
#include <float.h> //for FLT_EPSION
#include <algorithm>//for stable_sort, sort
using namespace std;


typedef struct NormalizedBBox_
{ //box
    float xmin;
    float ymin;
    float xmax;
    float ymax;
} NormalizedBBox;


void*  myAlloc(size_t size)
{

	char *ptr, *ptr0;
	//指针128对齐
	/*
	 * 假设要分配size字节内存。
	 * 基本思路就是先分配size+128-1字节的内存，
	 * 然后在起始的128字节里，找到128对齐的那个地址，作为对齐内存首地址，返回。
	 * 然后ptr0又多分配了一个sizeof(char*)字节的地址，在１２８对齐的ptr前８个字节里保存了ptr0的地址
	 *
	 */

	ptr0 = (char*)malloc(
		(size_t)(size + _MALLOC_ALIGN + sizeof(char*) -1 ));

	if (!ptr0)
		return 0;

	// align the pointer

    ptr = (char*)((size_t)(ptr0 + sizeof(char*) + _MALLOC_ALIGN - 1 ) & ~(size_t)(_MALLOC_ALIGN - 1));
    *(char**)(ptr - sizeof(char*)) = ptr0;

    return ptr;
}


void myFree_(void* ptr)
{
	// Pointer must be aligned by _MALLOC_ALIGN
	if (ptr)
	{
		if (((size_t)ptr & (_MALLOC_ALIGN - 1)) != 0)
			return;
		free(*((char**)ptr - 1));
	}

}

inline int dotProductUint8Int8(unsigned char * p1, signed char * p2, int num, int lengthInBytes)
{
    int sum = 0;

    for (int i = 0; i < num; i++)
    {
        sum += (int(p1[i]) * int(p2[i]));
    }
    return sum;
}

bool convolution1x1P0S1(const CDataBlob<unsigned char> *inputData, const Filters* filters, CDataBlob<int> *outputData)
{//卷积核为１＊１

    for (int row = 0; row < outputData->height; row++)
    {
        for (int col = 0; col < outputData->width; col++)
        {
            int * pOut = (outputData->data + (row*outputData->width + col)*outputData->channelStep / sizeof(int));
            unsigned char * pIn = (inputData->data + (row*inputData->width + col)*inputData->channelStep / sizeof(unsigned char));
            for (int ch = 0; ch < outputData->channels; ch++)
            {
                signed char * pF = (filters->filters[ch]->data);
                pOut[ch] = dotProductUint8Int8(pIn, pF, inputData->channels, inputData->channelStep);
            }
        }
    }
    return true;
}


bool convolution3x3P0(const CDataBlob<unsigned char> *inputData, const Filters* filters, CDataBlob<int> *outputData)
{   //卷积核为3＊3
    /*注意　num_pixels_inbytes是一行的三个元素，所以，只需要将与y轴向下移动三次就好了，
     * 这就是三个srcy＋＋块
     *
     *
     *
    */

    for (int row = 0; row < outputData->height; row++) 
    {  
        int elementStep = inputData->channelStep;
        int stride = filters->stride;
        int src_centery = row * stride;
        for (int col = 0; col < outputData->width; col++)
        {
            int srcx_start = col * stride - 1 ; //x纬度开始
            int srcx_end = srcx_start + 3 ; //x纬度截止
            srcx_start = MAX(0, srcx_start);
            srcx_end = MIN(srcx_end, inputData->width);
            int num_pixels_inbytes = (srcx_end - srcx_start) * elementStep; //所以每次乘都是每行的三个像素的，

            for (int ch = 0; ch < outputData->channels; ch++)
            {
                int srcy = src_centery - 1;

                unsigned char * pIn = (inputData->data + (srcy *inputData->width + srcx_start) * elementStep);
                signed char * pF = (filters->filters[ch]->data) + ( (srcx_start - col*stride + 1)) * elementStep;
                int * pOut = (outputData->data + (row*outputData->width + col)*outputData->channelStep / sizeof(int));
                pOut[ch] = 0;//the new created blob is not zeros, clear it first

                {
                    if (srcy >= 0)
                    {
                        pOut[ch] += dotProductUint8Int8(pIn,
                            pF,
                            num_pixels_inbytes,
                            num_pixels_inbytes);
                    }
                }
                {
                    srcy++;
                    {
                        pIn += (inputData->width * elementStep);
                        pOut[ch] += dotProductUint8Int8(pIn,
                            pF + (3 * elementStep),
                            num_pixels_inbytes,
                            num_pixels_inbytes);
                    }
                }
                {
                    srcy++;
                    if (srcy < inputData->height)
                    {
                        pIn += (inputData->width * elementStep);
                        pOut[ch] += dotProductUint8Int8(pIn,
                            pF + (6 * elementStep),
                            num_pixels_inbytes,
                            num_pixels_inbytes);
                    }
                }
            }
        }
    }
    return true; 
}


bool convolution(CDataBlob<unsigned char> *inputData, const Filters* filters, CDataBlob<int> *outputData) 
{
    if (inputData->data == NULL)
    {
        cerr << __FUNCTION__ << ": The input data is null." << endl;
        return false;
    }
    if (filters->filters.size() == 0)
    {
        cerr << __FUNCTION__ << ": There is not filters." << endl;
        return false;
    }
    /*Filters {
     * vector<CDataBlob<signed char> *> filters;
     * }
     */
    //check filters' dimensions
    //只支持步长为２或者1,padding 为１
    int filterW = filters->filters[0]->width;
    int filterH = filters->filters[0]->height;
    int filterC = filters->filters[0]->channels;
    int filterS = filters->stride;
    int filterP = filters->pad;

    int outputW = 0;
    int outputH = 0;
    int outputC = (int)filters->filters.size();
    /*这种情况不会存在
    for (int i = 1; i < outputC; i++)
    {
        if ((filterW != filters->filters[i]->width) ||
            (filterH != filters->filters[i]->height) ||
            (filterC != filters->filters[i]->channels))
        {
            cerr << __FUNCTION__ << ": The filters must be the same size." << endl;
            return false;
        }
    }*/

    if (filterC != inputData->channels)
    {
        cerr << __FUNCTION__ << ": The number of channels of filters must be the same with the input data. " << filterC << " vs " << inputData->channels << endl;
        return false;
    }

    //calculate the output dimension
    if (filterW == 1 && filterH == 1) //1x1 filters
    {
        if (filterS != 1)
        {
            cerr << __FUNCTION__ << ": Onle stride = 1 is supported for 1x1 filters." << endl;
            return false;
        }
        if (filterP != 0)
        {
            cerr << __FUNCTION__ << ": Onle pad = 0 is supported for 1x1 filters." << endl;
            return false;
        }
        outputW = inputData->width;
        outputH = inputData->height;

    }
    else if (filterW == 3 && filterH == 3) //3x3 filters
    {
        if (filterS == 1 && filterP == 1)
        {
            outputW = inputData->width;
            outputH = inputData->height;
        }
        else if (filterS == 2 && filterP == 1)
        {
            outputW = (inputData->width + 1) / 2;
            outputH = (inputData->height + 1) / 2;
        }
        else
        {
            cerr << __FUNCTION__ << ": Unspported filter stride=" << filterS << " or pad=" << filterP << endl;
            cerr << __FUNCTION__ << ": For 3x3 filters, only pad=1 and stride={1,2} are supported." << endl;
            return false;
        }
    }
    else
    {
        cerr << __FUNCTION__ << ": Unsported filter size." << endl;
        return false;
    }

    if (outputW < 1 || outputH < 1)
    {
        cerr << __FUNCTION__ << ": The size of the output is not correct. (" << outputW << ", " << outputH << ")." << endl;
        return false;
    }

    outputData->create(outputW, outputH, outputC);

    if (filterW == 1 && filterH == 1) //1x1 filters
    {
        convolution1x1P0S1(inputData, filters, outputData);
    }
    else if (filterW == 3 && filterH == 3) //3x3 filters
    {
        convolution3x3P0(inputData, filters, outputData);
    }

    outputData->scale = inputData->scale * filters->scale;

	return true;
}

bool convolution_relu(CDataBlob<unsigned char> *inputData, const Filters* filters, CDataBlob<unsigned char> *outputData)
{
    CDataBlob<int> tmpOutputData;  //注意这里是ｉｎｔ
    bool bFlag = convolution(inputData, filters, &tmpOutputData);
    if (bFlag == false)
        return false;

    //set negative values to zeros, 
    //and find the max value
    int nMaxValue = 0;


    for (int row = 0; row < tmpOutputData.height; row++)
    {
        for (int col = 0; col < tmpOutputData.width; col++)
        {
            int * pData = (tmpOutputData.data + (row*tmpOutputData.width + col)*tmpOutputData.channelStep / sizeof(int));

            for (int ch = 0; ch < tmpOutputData.channels; ch++)
            {
                pData[ch] = MAX(pData[ch], 0);
                nMaxValue = MAX(pData[ch], nMaxValue);
            }
        }
    }


    //scale the data to uint8 or int8
    float fCurrentScale = (_MAX_UINT8_VALUE) / float(nMaxValue);
    outputData->create(tmpOutputData.width, tmpOutputData.height, tmpOutputData.channels);
    outputData->scale = tmpOutputData.scale * fCurrentScale;

    for (int row = 0; row < outputData->height; row++)
    {
        for (int col = 0; col < outputData->width; col++)
        {
            int * pInt32Data = (tmpOutputData.data + (row*tmpOutputData.width + col)*tmpOutputData.channelStep / sizeof(int));
            unsigned char * pUInt8Data = (outputData->data + (row*outputData->width + col)*outputData->channelStep / sizeof(unsigned char));

            for (int ch = 0; ch < outputData->channels; ch++)
            {
                pUInt8Data[ch] = (unsigned char)(pInt32Data[ch] * fCurrentScale +0.499f);
            }
        }
    }
    return true;
}

//only 2X2 S2 is supported
bool maxpooling2x2S2(const CDataBlob<unsigned char> *inputData, CDataBlob<unsigned char> *outputData)
{
    if (inputData->data == NULL)
    {
        cerr << __FUNCTION__ << ": The input data is null." << endl;
        return false;
    }
    int outputW = static_cast<int>(ceil((inputData->width - 3.0f) / 2)) + 1;
    int outputH = static_cast<int>(ceil((inputData->height - 3.0f) / 2)) + 1;
    int outputC = inputData->channels;

    if (outputW < 1 || outputH < 1)
    {
        cerr << __FUNCTION__ << ": The size of the output is not correct. (" << outputW << ", " << outputH << ")." << endl;
        return false;
    }

    //int lineElementStep = inputData->width * inputData->channelStep;

    outputData->create(outputW, outputH, outputC);
    outputData->scale = inputData->scale;

    for (int row = 0; row < outputData->height; row++)
    {
        for (int col = 0; col < outputData->width; col++)
        {
            int inputMatOffsetsInElement[4];
            int elementCount = 0;

            int hstart = row * 2;
            int wstart = col * 2;
            int hend = MIN(hstart + 2, inputData->height);
            int wend = MIN(wstart + 2, inputData->width);

            for (int fy = hstart; fy < hend; fy++)
                for (int fx = wstart; fx < wend; fx++)
                {
                    inputMatOffsetsInElement[elementCount++] = (fy *inputData->width + fx) * inputData->channelStep / sizeof(unsigned char);
                }

            unsigned char * pOut = outputData->data + (row*outputData->width + col) * outputData->channelStep / sizeof(unsigned char);
            unsigned char * pIn = inputData->data;


            for (int ch = 0; ch < outputData->channels; ch++)
            {
                unsigned char maxval = pIn[ch + inputMatOffsetsInElement[0]];

                for (int el = 1; el < elementCount; el++)
                {
                    maxval = MAX(maxval, pIn[ch + inputMatOffsetsInElement[el]]);
                }
                pOut[ch] = maxval;
            }
        }
    }

    return true;
}


template<typename T>
bool concat4(const CDataBlob<T> *inputData1, const CDataBlob<T> *inputData2, const CDataBlob<T> *inputData3, const CDataBlob<T> *inputData4, CDataBlob<T> *outputData)
{
    if ((inputData1->data == NULL) || (inputData2->data == NULL) || (inputData3->data == NULL) || (inputData4->data == NULL))
    {
        cerr << __FUNCTION__ << ": The input data is null." << endl;
        return false;
    }

    if ((inputData1->width != inputData2->width) ||
        (inputData1->height != inputData2->height) ||
        (inputData1->width != inputData3->width) ||
        (inputData1->height != inputData3->height) ||
        (inputData1->width != inputData4->width) ||
        (inputData1->height != inputData4->height))
    {
        cerr << __FUNCTION__ << ": The three inputs must have the same size." << endl;
        return false;
    }
    int outputW = inputData1->width;
    int outputH = inputData1->height;
    int outputC = inputData1->channels + inputData2->channels + inputData3->channels + inputData4->channels;

    if (outputW < 1 || outputH < 1 || outputC < 1)
    {
        cerr << __FUNCTION__ << ": The size of the output is not correct. (" << outputW << ", " << outputH << ", " << outputC << ")." << endl;
        return false;
    }

    outputData->create(outputW, outputH, outputC);

    for (int row = 0; row < outputData->height; row++)
    {
        for (int col = 0; col < outputData->width; col++)
        {
            T * pOut = (outputData->data + (row*outputData->width + col)*outputData->channelStep / sizeof(T));
            T * pIn1 = (inputData1->data + (row*inputData1->width + col)*inputData1->channelStep / sizeof(T));
            T * pIn2 = (inputData2->data + (row*inputData2->width + col)*inputData2->channelStep / sizeof(T));
            T * pIn3 = (inputData3->data + (row*inputData3->width + col)*inputData3->channelStep / sizeof(T));
            T * pIn4 = (inputData4->data + (row*inputData4->width + col)*inputData4->channelStep / sizeof(T));

            memcpy(pOut, pIn1, sizeof(T)* inputData1->channels);
            memcpy(pOut + inputData1->channels, pIn2, sizeof(T)* inputData2->channels);
            memcpy(pOut + inputData1->channels + inputData2->channels, pIn3, sizeof(T)* inputData3->channels);
            memcpy(pOut + inputData1->channels + inputData2->channels + inputData3->channels, pIn4, sizeof(T)* inputData4->channels);
        }
    }
    return true;
}
template bool concat4(const CDataBlob<float> *inputData1, const CDataBlob<float> *inputData2, const CDataBlob<float> *inputData3, const CDataBlob<float> *inputData4, CDataBlob<float> *outputData);

bool convertInt2Float(CDataBlob<int> * inputData, CDataBlob<float> * outputData)
{
    if (inputData == NULL || outputData == NULL)
    {
        cerr << __FUNCTION__ << ": The input or output data is null." << endl;
        return false;
    }

    outputData->create(inputData->width, inputData->height, inputData->channels);
    float s = 1.0f / inputData->scale;
    for (int row = 0; row < outputData->height; row++)
    {
        for (int col = 0; col < outputData->width; col++)
        {
            int * pInData = (inputData->data + (row*inputData->width + col)*inputData->channelStep / sizeof(int));
            float * pOutData = (outputData->data + (row*outputData->width + col)*outputData->channelStep / sizeof(float));

            for (int ch = 0; ch < outputData->channels; ch++)
            {
                pOutData[ch] = pInData[ch] * s;
            }
        }
    }

    return true;
}

bool normalize(CDataBlob<unsigned char> * inputOutputData, float * pScale)
{
    if ((inputOutputData->data == NULL) || pScale == NULL)
    {
        cerr << __FUNCTION__ << ": The input data is null." << endl;
        return false;
    }

    CDataBlob<float> tmpData;
    tmpData.create(inputOutputData->width, inputOutputData->height, inputOutputData->channels);

    //normlize it,
    //and find the max value
    //because the input data is non-negative, so only the max value is needed
    float fMaxValue = 0;
    for (int row = 0; row < inputOutputData->height; row++)
    {
        for (int col = 0; col < inputOutputData->width; col++)
        {
            unsigned char * pInData = (inputOutputData->data + (row*inputOutputData->width + col)*inputOutputData->channelStep / sizeof(unsigned char));
            float * pTmpData = (float*)(tmpData.data + (row*tmpData.width + col)*tmpData.channelStep / sizeof(float));
            float sum = FLT_EPSILON;
            float s = 0;


            for (int ch = 0; ch < inputOutputData->channels; ch++)
                sum += ( float(pInData[ch]) * float(pInData[ch]));

            s = 1.0f / sqrt(sum);

            for (int ch = 0; ch < inputOutputData->channels; ch++)
            {
                pTmpData[ch] = pInData[ch] * pScale[ch] * s;  //pTmpData保存归一化的值
                fMaxValue = MAX(pTmpData[ch], fMaxValue);
            }
        }
    }

    //scale the data to uint8 or int8
    float fCurrentScale = (_MAX_UINT8_VALUE) / float(fMaxValue);
    inputOutputData->scale = fCurrentScale;

    for (int row = 0; row < inputOutputData->height; row++)
    {
        for (int col = 0; col < inputOutputData->width; col++)
        {
            float * pTmpData = (tmpData.data + (row*tmpData.width + col)*tmpData.channelStep / sizeof(float));
            unsigned char * pUInt8Data = (inputOutputData->data + (row*inputOutputData->width + col)*inputOutputData->channelStep / sizeof(unsigned char));

            for (int ch = 0; ch < inputOutputData->channels; ch++)
            {
                pUInt8Data[ch] = (unsigned char)(pTmpData[ch] * fCurrentScale + 0.499f);
            }
        }
    }

    return true;
}

bool priorbox(const CDataBlob<unsigned char> * featureData, const CDataBlob<unsigned char> * imageData, int num_sizes, float * pWinSizes, CDataBlob<float> * outputData)
{//只是根据featureData和imageData的size生成一些anchor_box并没有预测值
    if ((featureData->data == NULL) ||
        imageData->data == NULL||
        pWinSizes == NULL)
    {
        cerr << __FUNCTION__ << ": The input data is null." << endl;
        return false;
    }

    int feature_width = featureData->width;
    int feature_height = featureData->height;
    int image_width = imageData->width * 2;  //刚开始在create的时候尺寸缩小到一半，通道27
    int image_height = imageData->height * 2;

	float step_w = static_cast<float>(image_width) / feature_width;
	float step_h = static_cast<float>(image_height) / feature_height;

	//float * output_data = outputData->data;

    outputData->create(feature_width, feature_height, num_sizes * 4);

	for (int h = 0; h < feature_height; ++h) 
	{
		for (int w = 0; w < feature_width; ++w) 
		{
            float * pOut = (float*)(outputData->data + ( h * outputData->width + w) * outputData->channelStep / sizeof(float));
            int idx = 0;
            //priorbox
			for (int s = 0; s < num_sizes; s++) 
			{
				float min_size_ = pWinSizes[s];
                float box_width, box_height;
                box_width = box_height = min_size_;
                
                float center_x = w * step_w + step_w / 2.0f;
                float center_y = h * step_h + step_h / 2.0f;
                // xmin
                pOut[idx++] = (center_x - box_width / 2.f) / image_width;
                // ymin
                pOut[idx++] = (center_y - box_height / 2.f) / image_height;
                // xmax
                pOut[idx++] = (center_x + box_width / 2.f) / image_width;
                // ymax
                pOut[idx++] = (center_y + box_height / 2.f) / image_height;

			}
		}
	}
    
    return true;
}

bool softmax1vector2class(CDataBlob<float> *inputOutputData)
{
    if (inputOutputData == NULL )
    {
        cerr << __FUNCTION__ << ": The input data is null." << endl;
        return false;
    }

    if(inputOutputData->width != 1 || inputOutputData->height != 1)
    {
        cerr << __FUNCTION__ << ": The input data must be Cx1x1." << endl;
        return false;
    }

    int num = inputOutputData->channels;
    float * pData = inputOutputData->data;

    for(int i = 0; i < num; i+= 2)
    {
        float v1 = pData[i];
        float v2 = pData[i + 1];
        float vm = MAX(v1, v2);
        v1 -= vm;
        v2 -= vm;
        v1 = exp(v1);  //expf在１１的标准上没有
        v2 = exp(v2);
        vm = v1 + v2;
        pData[i] = v1/vm;
        pData[i+1] = v2/vm;
    }
    return true;
}

template<typename T>
bool blob2vector(const CDataBlob<T> * inputData, CDataBlob<T> * outputData)
{   //将矩阵变为一维矩阵即安装通道，列，行的排列
    //inputData是128对齐的，且通道数也是对齐的
    //outputData仅仅是128对齐
    //但是outputData通过create后就是两个都对齐了，如果outputData不用对齐后的通道，那就是浪费了空间了

    if (inputData->data == NULL || outputData == NULL)
    {
        cerr << __FUNCTION__ << ": The input or output data is null." << endl;
        return false;
    }

    outputData->create(1, 1, inputData->width * inputData->height * inputData->channels);
    outputData->scale = inputData->scale;

    int bytesOfAChannel = inputData->channels * sizeof(T);
    T * pOut = outputData->data;
    for (int row = 0; row < inputData->height; row++)
    {
        for (int col = 0; col < inputData->width; col++)
        {
            T * pIn = (inputData->data + (row*inputData->width + col)*inputData->channelStep / sizeof(T));
            memcpy(pOut, pIn, bytesOfAChannel);
            pOut += inputData->channels;

        }
    }

    return true;
}
template bool blob2vector(const CDataBlob<signed char> * inputData, CDataBlob<signed char> * outputData);
template bool blob2vector(const CDataBlob<int> * inputData, CDataBlob<int> * outputData);
template bool blob2vector(const CDataBlob<float> * inputData, CDataBlob<float> * outputData);

void IntersectBBox(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
                   NormalizedBBox* intersect_bbox) 
{//俩两个box的交叉部分
    if (bbox2.xmin > bbox1.xmax || bbox2.xmax < bbox1.xmin ||
        bbox2.ymin > bbox1.ymax || bbox2.ymax < bbox1.ymin) 
    {
        // Return [0, 0, 0, 0] if there is no intersection.
        intersect_bbox->xmin = 0;
        intersect_bbox->ymin = 0;
        intersect_bbox->xmax = 0;
        intersect_bbox->ymax = 0;
    }
    else
    {
        intersect_bbox->xmin = (std::max(bbox1.xmin, bbox2.xmin));
        intersect_bbox->ymin = (std::max(bbox1.ymin, bbox2.ymin));
        intersect_bbox->xmax = (std::min(bbox1.xmax, bbox2.xmax));
        intersect_bbox->ymax = (std::min(bbox1.ymax, bbox2.ymax));
    }
}

float JaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2)
{// IOU Value

    NormalizedBBox intersect_bbox;
    IntersectBBox(bbox1, bbox2, &intersect_bbox);
    float intersect_width, intersect_height;
    intersect_width = intersect_bbox.xmax - intersect_bbox.xmin;
    intersect_height = intersect_bbox.ymax - intersect_bbox.ymin;

    if (intersect_width > 0 && intersect_height > 0) 
    {
        float intersect_size = intersect_width * intersect_height;
        float bsize1 = (bbox1.xmax - bbox1.xmin)*(bbox1.ymax - bbox1.ymin);
        float bsize2 = (bbox2.xmax - bbox2.xmin)*(bbox2.ymax - bbox2.ymin);
        return intersect_size / ( bsize1 + bsize2 - intersect_size);
    }
    else 
    {
        return 0.f;
    }
}

bool SortScoreBBoxPairDescend(const pair<float, NormalizedBBox>& pair1,   const pair<float, NormalizedBBox>& pair2) 
{
    return pair1.first > pair2.first;
}


bool detection_output(const CDataBlob<float> * priorbox, const CDataBlob<float> * loc, const CDataBlob<float> * conf, float overlap_threshold, float confidence_threshold, int top_k, int keep_top_k, CDataBlob<float> * outputData)
{
    if (priorbox->data == NULL || loc->data == NULL || conf->data == NULL)
    {
        cerr << __FUNCTION__ << ": The input data is null." << endl;
        return 0;
    }

    if (priorbox->channels != loc->channels || loc->channels != conf->channels*2 )
    {
        cerr << __FUNCTION__ << ": The sizes of the inputs are not match." << endl;
        cerr << "priorbox channels=" << priorbox->channels << ", loc channels=" << loc->channels << ", conf channels=" << conf->channels << endl;
        return 0;
    }

    float prior_variance[4] = {0.1f, 0.1f, 0.2f, 0.2f};
    float * pPriorBox = priorbox->data;
    float * pLoc = loc->data;
    float * pConf = conf->data;

    vector<pair<float, NormalizedBBox> > score_bbox_vec;
    vector<pair<float, NormalizedBBox> > final_score_bbox_vec;

    //get the candidates those are > confidence_threshold
    for(int i = 1; i < conf->channels; i+=2)
    {
        if(pConf[i] > confidence_threshold)
        {
            float fx1 = pPriorBox[i*2-2];  //min_x
            float fy1 = pPriorBox[i*2-1];  //min_y
            float fx2 = pPriorBox[i*2];  //max_x
            float fy2 = pPriorBox[i*2+1];  //max_y

            float locx1 = pLoc[i * 2 - 2];
            float locy1 = pLoc[i * 2 - 1];
            float locx2 = pLoc[i * 2];
            float locy2 = pLoc[i * 2 + 1];

            float prior_width = fx2 - fx1;
            float prior_height = fy2 - fy1;
            float prior_center_x = (fx1 + fx2)/2;
            float prior_center_y = (fy1 + fy2)/2;

            float box_centerx = prior_variance[0] * locx1 * prior_width + prior_center_x;
            float box_centery = prior_variance[1] * locy1 * prior_height + prior_center_y;
            float box_width = exp(prior_variance[2] * locx2) * prior_width;
            float box_height = exp(prior_variance[3] * locy2) * prior_height;
            //float box_width = expf(prior_variance[2] * locx2) * prior_width;  //expf在１１的标准上没有
           // float box_height = expf(prior_variance[3] * locy2) * prior_height;

            fx1 = box_centerx - box_width / 2.f;
            fy1 = box_centery - box_height /2.f;
            fx2 = box_centerx + box_width / 2.f;
            fy2 = box_centery + box_height /2.f;

            fx1 = MAX(0, fx1);
            fy1 = MAX(0, fy1);
            fx2 = MIN(1.f, fx2);
            fy2 = MIN(1.f, fy2);

            NormalizedBBox bb;
            bb.xmin = fx1;
            bb.ymin = fy1;
            bb.xmax = fx2;
            bb.ymax = fy2;

            score_bbox_vec.push_back(std::make_pair(pConf[i], bb));
        }
    }

    //Sort the score pair according to the scores in descending order
    std::stable_sort(score_bbox_vec.begin(), score_bbox_vec.end(), SortScoreBBoxPairDescend);  //也可以使用ｓｏｒｔ


    // Keep top_k scores if needed.
    if (top_k > -1 && top_k < score_bbox_vec.size()) {
        score_bbox_vec.resize(top_k);  //保留前top_k个
    }

    //Do NMS
    final_score_bbox_vec.clear();
    while (score_bbox_vec.size() != 0) {
        const NormalizedBBox bb1 = score_bbox_vec.front().second;  //second是因为make_pair 将pConf和bb打包了
        bool keep = true;
        for (int k = 0; k < final_score_bbox_vec.size(); ++k)
        {
            if (keep) 
            {
                const NormalizedBBox bb2 = final_score_bbox_vec[k].second;
                float overlap = JaccardOverlap(bb1, bb2);
                keep = (overlap <= overlap_threshold);
            }
            else 
            {
                break;
            }
        }
        if (keep) {  //针对最大的那个值，即排序后的第一个
            final_score_bbox_vec.push_back(score_bbox_vec.front());
        }
        score_bbox_vec.erase(score_bbox_vec.begin());
    }
    if (keep_top_k > -1 && keep_top_k < final_score_bbox_vec.size()) {
        final_score_bbox_vec.resize(keep_top_k);
    }

    //copy the results to the output blob
    int num_faces = (int)final_score_bbox_vec.size();
    if (num_faces == 0)
        outputData->setNULL();
    else
    {
        outputData->create(num_faces, 1, 5);
        for (int fi = 0; fi < num_faces; fi++)
        {
            pair<float, NormalizedBBox> pp = final_score_bbox_vec[fi];
            float * pOut = (outputData->data + fi * outputData->channelStep / sizeof(float));
            pOut[0] = pp.first; //cof
            pOut[1] = pp.second.xmin;
            pOut[2] = pp.second.ymin;
            pOut[3] = pp.second.xmax;
            pOut[4] = pp.second.ymax;
        }
    }

    return true;
}

