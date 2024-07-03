/*
입력 이미지 데이터를 읽어와 추론을 통해 결과를 도출
*/


#include <stdio.h>
#include <math.h>
#include "inference.h"    // 함수 선언과 상수들을 사용하기 위해 포함

void softmax (float *p, float * softmax_result, unsigned int n);
//프로그램의 
int main(int argc, char *argv[]) { 
    
//////////////////////////////////////////////////////////////////////
// Read image data
//////////////////////////////////////////////////////////////////////

    if(argc != 2){
        printf("Usage %s input_image\n", argv[0]);   //현재 실행되는 파일의 위치 알려줌
    }

    // Read image data
    float DataFloat [IMG_CHANNELS * IMG_DMNIN * IMG_DMNIN]; //이미지 데이터를 저장할 DataFloat 배열을 선언 , IMG_CHANNELS는 이미지 개수, IMG_DMNIN은 이미지의 크기
    float result[NUM_CLASSES]; // 최종 결과를 저장할 result 배열을 선언
    float resultClasses[NUM_CLASSES]; 

    if(get_image_data(argv[1], &DataFloat[0]) == -1){
        return -1;
    }

//////////////////////////////////////////////////////////////////////
// End of Read image data
//////////////////////////////////////////////////////////////////////

    inference(DataFloat, result);
    softmax(result,resultClasses, NUM_CLASSES);
    /*
    //Print the output
    printf("Output:\n");
    for (int f = 0; f < C1_N_FILTERS; f++) {
    printf("Filter %d:\n", f + 1);
    for (int i = 0; i < C1_OUT_DMNIN; i++) {
    for (int j = 0; j < C1_OUT_DMNIN; j++) {
    printf("%.2f ", *((float*)result+ (f * C1_OUT_DMNIN * C1_OUT_DMNIN) + (i * C1_OUT_DMNIN) + j));
            }  
    printf("\n");
        }
    }
    */
    return 0;
}

void softmax (float *p, float * softmax_result, unsigned int n)
{
     float denom = 0.0f;
    
    // Calculate sum of exponentials of input values
    for (unsigned int i = 0; i < n; i++) {
        float fval = p[i];
        denom += expf(fval);
    }
    
    // Calculate softmax values
    for (unsigned int i = 0; i < n; i++) {
        float fval = p[i];
        softmax_result[i] = expf(fval) / denom;
        printf("%d: %.1f\n",i,softmax_result[i]);
    }
}