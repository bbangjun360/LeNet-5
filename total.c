#include "inference.h"
#include "biases.h"
#include "weights.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// inference 함수는 이미지를 입력받아 신경망 추론을 수행하고 결과를 반환
void inference(float *img, float *p_result)
{
    float layer1_out[C1_N_FILTERS][C1_OUT_DMNIN][C1_OUT_DMNIN];
    float layer2_out[C1_N_FILTERS][C1_OUT_DMNIN][C1_OUT_DMNIN];
    float layer3_out[C1_N_FILTERS][P1_DOWNSIZE][P1_DOWNSIZE];
    float layer4_out[C2_N_FILTERS][C2_OUT_DMNIN][C2_OUT_DMNIN]; //
    float layer5_out[C2_N_FILTERS][C2_OUT_DMNIN][C2_OUT_DMNIN];
    float layer6_out[C2_N_FILTERS][P2_DOWNSIZE][P2_DOWNSIZE];
    float layer7_out[F1_ROWS];
    float layer8_out[F1_ROWS];
    float layer9_out[F2_ROWS];
    float layer10_out[F2_ROWS];
    //float p_result[F3_ROWS];

    //입력 이미지를 첫 번째 컨볼루션 레이어와 ReLU활성화 함수를 통과
    conv_3D((float*) img, (const float*) weights_C1, (float*) layer1_out, 
            (const float*) biases_C1, C1_N_CHAN, C1_N_FILTERS, C1_X_DMNIN, C1_W_DMNIN, C1_OUT_DMNIN, STRIDE, PADDING);
    relu_3D((float*)layer1_out, (float *)layer2_out, C1_N_FILTERS, C1_OUT_DMNIN, C1_OUT_DMNIN);
    max_pooling((float *)layer2_out, (float *)layer3_out, C1_N_FILTERS, P1_DOWNSIZE);

    conv_3D((float*) layer3_out, (const float*) weights_C2, (float*) layer4_out, 
           (const float*) biases_C2, C2_N_CHAN, C2_N_FILTERS, C2_X_DMNIN, C2_W_DMNIN, C2_OUT_DMNIN, STRIDE, PADDING);
    relu_3D((float*)layer4_out, (float *)layer5_out, C2_N_FILTERS, C2_OUT_DMNIN, C2_OUT_DMNIN);
    max_pooling((float *)layer5_out, (float *)layer6_out, C2_N_FILTERS, P2_DOWNSIZE);

    full_connection((float *)layer6_out, (const float *)weights_F1, (const float *)biases_F1, 
                    (float *)layer7_out, FLAT_VEC_SZ, F1_ROWS);
    relu_1D((float *)layer7_out, (float *)layer8_out, F1_ROWS);

    full_connection((float *)layer8_out, (const float *)weights_F2, (const float *)biases_F2, 
             (float *)layer9_out, F1_ROWS, F2_ROWS);

    relu_1D((float *)layer9_out, (float *)layer10_out, F2_ROWS);

    full_connection((float *)layer10_out, (const float *)weights_F3, (const float *)biases_F3, 
             (float *)p_result, F2_ROWS, F3_ROWS);


}


// 이미지 파일을 로드하고 정규화하여 신경망에 입력으로 사용할 수 있도록 한다
int get_image_data(char inputFileName[], float *pImg_float)
{

    int width, height, channels;

    unsigned char *pImg = stbi_load(inputFileName, &width, &height, &channels, 0);

    if(pImg == NULL) return -1;

    if(!(width == 28 && height==28 && channels==1)){
        puts("Input image size is not correct,");
        printf("%s() %s w=%d h=%d c=%d\n", __func__, inputFileName, width, height, channels);
        return -1;
    }
    printf("%s w=%d h=%d c=%d\n", inputFileName, width, height, channels);
//----------------------------------------2부---------------------------------------------------------------
    // zero padding - 28 - 32 
    // 32x32 pImg_float 배열을 0.0f로 전체 초기화
    for(int h=0; h<IMG_DMNIN; h++)
        for(int w=0; w<IMG_DMNIN; w++)
            *(pImg_float + w + h*IMG_DMNIN) = 0.0f;

    for(int h=0; h<height; h++){
        for(int w=0; w<width; w++){
            // normalization
            *(pImg_float + (h+2)*32 + (w+2)) = *(pImg + h*width + w) / 255.0;
        }
    }
    stbi_image_free(pImg);

    return 0;
    
}

//3D 컨볼루션을 수행
void conv_3D(
    float* input, // 
    const float* weights, 
    float* output, 
    const float* biases, 
    int c_n_chan, int c_n_filters, int c_x_dmnin, int c_w_dmnin, int c_out_dmnin, //c_n_chan : input 갯수 1개, c_n_filters : filter 갯수 6개, c_x_dmnin : input사진 크기 32, c_w_dmnin : filter 사이즈, c_out_dmnin : convolution 완료된 feature의 크기
    int stride, int padding)
{
    
    //Print the input value
    printf("input: \n");
    for(int i=0; i< IMG_CHANNELS; i++)
    {
        for(int j=0; j<IMG_DMNIN; j++)
        {
            for(int k=0; k<IMG_DMNIN; k++)
                printf("%.1f ", input[k+j*IMG_DMNIN+i*IMG_DMNIN*IMG_DMNIN]);
            printf("\n");
        }
        printf("\n\n");
    }

    //Print the filter value
    printf("filter: \n");
    for (int filter = 0; filter < C1_N_FILTERS; filter++) {
        for (int channel = 0; channel < C1_N_CHAN; channel++) {
            for (int i = 0; i < C1_W_DMNIN; i++) {
                for (int j = 0; j < C1_W_DMNIN; j++) {
                    printf("%f ", weights[j+i*C1_W_DMNIN+channel*C1_W_DMNIN*C1_W_DMNIN+filter*C1_N_CHAN*C1_W_DMNIN*C1_W_DMNIN]);
                }
                printf("\n");
            }
            printf("\n\n");
        }
        printf("\n\n\n");
    }
    

    // Convolution //
    // Loop for Filter
    printf("convolution: \n");
    for(int f=0; f<c_n_filters; f++)    // filter # == output channel #
    {
        // Loop for Output 
        printf("filter %d: \n",f);
        for(int i=0; i<c_out_dmnin; i++)    // output row
        {
            for(int j=0; j<c_out_dmnin; j++)    // output col
            {
                float weighted_sum = 0;   // Initialize the weighted sum

                // Loop for Input Channel
                for(int ch=0; ch<c_n_chan; ch++)    // input channel # == filter channel #
                {
                    // Loop for Kernel Row
                    for(int ki=0; ki<c_w_dmnin; ki++)   // 
                    {
                        // Loop for Kernel Col
                        for(int kj=0; kj<c_w_dmnin; kj++)
                        {
                            // index for input, weights considering stride and padding
                            int i_row = i*stride + ki - padding;
                            int i_col = j*stride + kj - padding;

                            if((i_row >= 0) && (i_row < c_x_dmnin) && (i_col >= 0) && (i_col < c_x_dmnin))
                            {
                                int i_idx = i_col + (i_row*c_x_dmnin) + (ch*c_x_dmnin*c_x_dmnin);
                                int w_idx = kj + (ki*c_w_dmnin) + (ch*c_w_dmnin*c_w_dmnin) + (f*(c_w_dmnin*c_w_dmnin*c_n_chan));
                                
                                // MAC
                                weighted_sum += (*(input+i_idx)) * (*(weights+w_idx));
                            }
                        }
                    }
                }
                // index for output
                int o_idx = j + (i*c_out_dmnin) + (f*c_out_dmnin*c_out_dmnin);
                
                // output
                output[o_idx] = weighted_sum + biases[f];
                printf("%.1f ", output[o_idx]);

            }
            printf("\n");
        }
        printf("\n\n");
    }
    
}
void relu_1D(float* input, float* output, int rows){
    printf("RELU 1D: \n");
    for(int r=0; r<rows; r++)
    {
        output[r] = input[r] > 0 ? input[r] : 0;
        printf("%.1f ", output[r] );
    }
    printf("\n\n");
}
//ReLU 활성화 함수 적용
void relu_3D(float* input, float* output, int c_n_filters, int c_out_dmnin1, int c_out_dmnin2)
{
    printf("relu 3d:\n");
    for(int ch=0; ch<c_n_filters; ch++)
    {
        printf("filter %d \n",ch);
        for(int w=0; w<c_out_dmnin2; w++)
        {
            for(int h=0; h<c_out_dmnin1; h++)
            {
                output[h + w*c_out_dmnin1 + ch*c_out_dmnin1*c_out_dmnin2] 
                = input[h + w*c_out_dmnin1 + ch*c_out_dmnin1*c_out_dmnin2] > 0 ?
                    input[h + w*c_out_dmnin1 + ch*c_out_dmnin1*c_out_dmnin2] : 0;
                printf("%.3f ", output[h + w*c_out_dmnin1 + ch*c_out_dmnin1*c_out_dmnin2] );
            }
            printf("\n");
        }
        printf("\n\n");
    }
}

// S2 (subsampling_maxpooling) 함수 적용
void max_pooling(float* input, float* output, int c_n_filters, int p1_downsize)
{
    // Max Pooling 연산 수행 
    printf("subsamping S2:\n");
    for(int ch=0; ch< c_n_filters; ch++)
    {
        printf("subsamping ch: %d\n", ch);
        for (int i = 0; i < p1_downsize; i++) {
            for (int j = 0; j < p1_downsize; j++) 
            {
                // 필터 내 왼쪽, 위쪽에 위치한 인덱스 계산
                int r = i * MAXPOOLING_STRIDE;
                int c = j * MAXPOOLING_STRIDE;
                int ch_idx = ch * C1_OUT_DMNIN * C1_OUT_DMNIN;

                // 각 필터 내에서 최댓값 찾기 
                float max_value = *(input + r * C1_OUT_DMNIN + c + ch_idx);
                max_value = (*(input + r * C1_OUT_DMNIN + (c + 1) + ch_idx) > max_value) ? *(input + r * C1_OUT_DMNIN + (c + 1) + ch_idx) : max_value;
                max_value = (*(input + (r + 1) * C1_OUT_DMNIN + c + ch_idx) > max_value) ? *(input + (r + 1) * C1_OUT_DMNIN + c + ch_idx) : max_value;
                max_value = (*(input + (r + 1) * C1_OUT_DMNIN + (c + 1) + ch_idx) > max_value) ? *(input + (r + 1) * C1_OUT_DMNIN + (c + 1) + ch_idx) : max_value;

                // 출력 배열에 최댓값 저장
                *(output + i * p1_downsize + j + (ch * p1_downsize * p1_downsize)) = max_value;
                printf("%.1f ", max_value);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void full_connection(float* input, const float* weights_f1, const float* biases_f1, float* output, int flat_vec_sz, int f1_rows)
{
    printf("full connection: \n");
    
    for (int i = 0; i < f1_rows; ++i) {
        float sum = 0.0;
        for (int j = 0; j < flat_vec_sz; ++j)
            sum += (*(input + j)) * (*(weights_f1 + (i * flat_vec_sz) + j));
        *(output + i) = sum + *(biases_f1 + i); // 편향 값 수정
        printf("%0.1f ", *(output + i)); // 출력된 값 확인
    }
    printf("\n\n");
}

