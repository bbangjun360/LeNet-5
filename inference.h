#ifndef INFERENCE_H
#define INFERENCE_H

#define NUM_CLASSES     10

#define IMG_DMNIN       32 //zero padding 한 이미지의 크기
#define IMG_CHANNELS    1 //받은 이미지의 개수
#define STRIDE          1
#define PADDING         0 //stride의 패딩 = 0

//CONVOLUTION #1
#define C1_N_CHAN       1 //c1에서 들어오는 입력 이미지의 개수
#define C1_N_FILTERS    6 //c1 convolution에서 fiter 개수
#define C1_X_DMNIN      32 //c1 convolution 들어온 이미지의 크기
#define C1_W_DMNIN      5 // c1 convolution에서 filter의 크기
#define C1_OUT_DMNIN    28 // c1 convolution의 결과 이미지의 크기

//MAXPOOLING #1
#define P1_DOWNSIZE     14

//CONVOLUTION #2
#define C2_N_CHAN       6
#define C2_N_FILTERS    16
#define C2_X_DMNIN      14
#define C2_W_DMNIN      5
#define C2_OUT_DMNIN    10

//MAXPOOLING #2
#define MAXPOOLING_STRIDE 2 //맥스 풀링 스트라이드 2

#define P2_DOWNSIZE     5

#define FLAT_VEC_SZ     400

#define F1_ROWS         120

#define F2_ROWS         84

#define F3_ROWS         10


int get_image_data(char inputFileName[], float *pImg_float);
//void conv_3D(float* input, const float* weights, float* output, const float* biases, int c_n_chan, int c_n_filters, int c_x_dmnin, int c_w_dmnin, int c_out_dmnin);
void conv_3D(float* input, const float* weights, float* output, const float* biases, int c_n_chan, int c_n_filters, int c_x_dmnin, int c_w_dmnin, int c_out_dmnin, int stride, int padding);
void relu_1D(float* input, float* output, int rows);
void relu_3D(float* input, float* output, int c_n_filters, int c_out_dmnin1, int c_out_dmnin2);
void inference(float *img, float *p_result);
void max_pooling(float* input, float* output, int c_n_filters, int p_downsize);
void full_connection(float* input, const float* weights_f1, const float* biases_f1, float* output, int flat_vec_sz, int f1_rows);
#endif

