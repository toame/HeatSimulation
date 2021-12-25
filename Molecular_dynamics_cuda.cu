#include <iostream>
#include <random>
#include <cmath>
#include <chrono>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <cufft.h>
#include <cassert>
#define blockSize (64)
#define mu1 (1)
#define mu1_bath (1)
#define mu0 (0)
#define beta0 (1)
#define dt (0.05)              // 時間刻み幅
#define dt_1 (1.0 / dt)
#define dt2 (dt * dt)
#define gamma0 (0.2)          // Langevin熱浴のgamma係数
#define gamma_t (gamma0 * dt)   
#define statistics_interval (1)

#ifdef _WIN32
    #define SLASH "\\"
#else
    #define SLASH "/"
#endif
using namespace std;

__global__ 
void setCurand(unsigned long long seed, curandState *state){
    unsigned int i_global = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, i_global, 0, &state[i_global]);
}
__global__
void qinit(float *q1, float *q0, const int size_)
{
    const int thread_id = threadIdx.x+blockDim.x*blockIdx.x;
    const int i = thread_id;
    if(i < size_) {
        q0[i] = q1[i] = 0.0;
    }
}
__global__
void statistics_reset_d(float* fluxC, const int model_size) {
    const int thread_id = threadIdx.x+blockDim.x*blockIdx.x;
    const int i = thread_id + 1;
    if(i < model_size)
        fluxC[i] = 0.0;
}

__global__
void Update(float* q2, const float* q1, const float *q0,curandState *state,  float* ct_c, float *fluxC,float *temperature_plot, const int n1_L, const int n1_R, const int n2_L, const int n2_R, const int n3_L, const int n3_R, const bool is_heat)
{
    const int thread_id = threadIdx.x+blockDim.x*blockIdx.x;
    const int i = thread_id + 1;

    const int is_middleL = (n2_L < i && i <= n2_R);
    const int is_middleR = (n2_L <= i && i < n2_R);
    
    const float muL = is_middleL * mu1 + !(is_middleL) * mu1_bath;
    const float muR = is_middleR * mu1 + !(is_middleR) * mu1_bath;

    const float f_R = muR * (q1[i + 1] - q1[i]);
    const float f_L = muL * (q1[i] - q1[i - 1]);
    const float l_f = f_R - f_L;

    const float nf_R = (is_middleR) * beta0 * (q1[i + 1] - q1[i]) * (q1[i + 1] - q1[i]) * (q1[i + 1] - q1[i]);
    const float nf_L = (is_middleL) * beta0 * (q1[i - 1] - q1[i]) * (q1[i - 1] - q1[i]) * (q1[i - 1] - q1[i]);
    const float nl_f = nf_L + nf_R;

    const float dq1 = q1[i] - q0[i];
    const float random_f = (is_heat || i <= n1_R || i >= n3_L) ? (-gamma_t * dq1 + ct_c[i] * curand_normal(&state[i])) : 0;

    const float f = l_f + nl_f - mu0 * q1[i];
    const float dq2 = dq1 + f * dt2 + random_f;  

    q2[i] = q1[i] + dq2;

    // 統計量の計算
    const float p = 0.5 * (q2[i] - q0[i]) * dt_1;
    fluxC[i] += p * (-f_L + nf_L);
    temperature_plot[i] += p * p;
}

class FPUT_Lattice_1D{
private:
    int HeatBath_size;                      // 熱浴サイズ
    int middle_size;                        // 調べたいモデルサイズ
    int n1_L;                               // 高温熱浴の左端のインデックス
    int n1_R;                               // 高温熱浴の右端のインデックス
    int n2_L;                               // FPUT_Latticeの左端のインデックス
    int n2_R;                               // FPUT_Latticeの右端のインデックス
    int n3_L;                               // 低温熱浴の左端のインデックス
    int n3_R;                               // 低温熱浴の右端のインデックス
    int model_size;                         // モデル全体の粒子数

    long long int HeatSimulation;           // モデル全体に熱浴をかける（収束を早めたいため）
    long long int initialStateStep;         // 統計量を測定開始するステップ数（最初は収束していないため）
    long long int Step;                     // 全体のステップ数

    const float temp_h = 1.2;              // 高温熱浴の温度
    const float temp_l = 0.8;              // 低温熱浴の温度

    const float c_h = sqrt(2.0*gamma0*temp_h*dt);  // 高温側の分散
    const float c_l = sqrt(2.0*gamma0*temp_l*dt);  // 低温側の分散
    
    const float ct_h = c_h * dt;           
	const float ct_l = c_l * dt;        
    float* h_ct_c;           // 中央部分の（初期）温度  
    float* d_ct_c;           // 中央部分の（初期）温度    

    float* q0;                         // 2ステップ前の粒子位置
    float* q1;                         // 1ステップ前の粒子位置
    float* q2;                         // 現在の粒子位置    
    float* h_fluxC;
    float* h_temperature_plot;
    float* d_fluxC;
    float* d_temperature_plot;

    long long int count_flux = 0;
    long long int count_temp = 0;
    long long int stepCount = 0;

    curandState *state;   
public:
    
    FPUT_Lattice_1D() {};

    void settingSize(const int HeatBath_size_, const int middle_size_) {
        HeatBath_size = HeatBath_size_;
        middle_size = middle_size_;
        n1_L = 1;
        n1_R = n1_L + HeatBath_size - 1;
        n2_L = n1_R + 1;
        n2_R = n2_L + middle_size - 1;
        n3_L = n2_R + 1;
        n3_R = n3_L + HeatBath_size - 1;
        model_size = n3_R + 2;
        cerr << n1_L << "," << n1_R << "," << n2_L << "," << n2_R << "," << n3_L << "," << n3_R <<"," << model_size <<  endl;
        h_ct_c = (float *)malloc(sizeof(float)* (model_size + 256));
        h_fluxC = (float *)malloc(sizeof(float) * (model_size + 256));
        h_temperature_plot = (float *)malloc(sizeof(float) * (model_size + 256));
        cudaMalloc(&d_ct_c, sizeof(float) * (model_size + 256));
        cudaMalloc(&d_fluxC, sizeof(float) * (model_size + 256));
        cudaMalloc(&d_temperature_plot, sizeof(float) * model_size);

        cudaMalloc(&q0, sizeof(float) * (model_size + 256));
        cudaMalloc(&q1, sizeof(float) * (model_size + 256));
        cudaMalloc(&q2, sizeof(float) * (model_size + 256));
        cudaMalloc(&state,  sizeof(float) * (model_size + 256) * sizeof(curandState));
        for(int i = n1_L; i <= n1_R; i++) {
            h_ct_c[i] = ct_h;
        }
        for(int i = n2_L; i <= n2_R; i++) {
            const float temp_c = temp_h - (temp_h - temp_l) * (i - n2_L)/(n2_R - n2_L + 1);
            const float c_c = sqrt(2.0*gamma0*temp_c*dt); 
            h_ct_c[i] = c_c * dt;
        }
        for(int i = n3_L; i <= n3_R; i++) {
            h_ct_c[i] = ct_l;
        }
        cudaMemcpy(d_ct_c, h_ct_c, sizeof(float) * model_size, cudaMemcpyHostToDevice);
        dim3 grid_size = dim3(model_size / blockSize + 1, 1, 1);
        dim3 block_size = dim3(blockSize, 1, 1);
        setCurand<<<grid_size,block_size>>>(1, state);
        qinit<<<grid_size,block_size>>>(q0, q1, model_size);
    }
    void settingStep(const long long int HeatSimulation_, const long long int initialStateStep_, const long long int Step_) {
        HeatSimulation = HeatSimulation_;
        initialStateStep = initialStateStep_;
        Step = Step_;
    }
    // 進行度の出力
    void showProcessing() {
        // end = std::chrono::system_clock::now();
        // float time = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 / 1000.0);
        // std::cerr << "process = " << std::fixed << std::setprecision(3) << (float)stepCount/(float)allSteps * 100.0 << "%, ";
        // std::cerr << "StepCount = " << std::setfill('0') << std::right << std::setw(9) << stepCount << "/" << std::setfill('0') << std::right << std::setw(9) << allSteps << ", ";
        // std::cerr << "t = " << std::fixed << std::setprecision(6) << time/86400.0 << " [days], ";
        // std::cerr << "est. = " << std::fixed << std::setprecision(6) << time/((float)stepCount/(float)allSteps)/86400.0<< " [days]" << std::endl;
        float *h_q0L = (float*) malloc(sizeof(float));
        float *h_q0M = (float*) malloc(sizeof(float));
        float *h_q0R = (float*) malloc(sizeof(float));
        cudaMemcpy(h_q0L, q0 + (n1_L + n1_R) / 2, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_q0M, q0 + middle_size / 2, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_q0R, q0 + (n3_L + n3_R) / 2, sizeof(float), cudaMemcpyDeviceToHost);
        std::cerr << *h_q0L << " " << *h_q0M << " " << *h_q0R << std::endl;
        assert(*h_q0M < 1e+10);

        free(h_q0L); free(h_q0M); free(h_q0R);
    }

    void step() {
        //void Update(float *q2, float* q1, float *q0, float ct_c, float *fluxC, curandState *state, float *temperature_plot, const int n1_L, const int n1_R, const int n2_L, const int n2_R, const int n3_L, const int n3_R, const bool is_heat, const float *f_nl)
        dim3 grid_size = dim3(model_size / blockSize + 1, 1, 1);
        dim3 block_size = dim3(blockSize, 1, 1);
        Update<<<grid_size,block_size>>>(q2, q1, q0, state,d_ct_c, d_fluxC, d_temperature_plot, n1_L, n1_R, n2_L, n2_R, n3_L, n3_R, stepCount < HeatSimulation);
        swap(q0, q1);
        swap(q1, q2);
        count_flux += statistics_interval;
        count_temp += statistics_interval;
        stepCount += statistics_interval;
    }
    void statistics_reset() {
        count_flux = 0;
        count_temp = 0;
        dim3 grid_size = dim3(model_size / blockSize + 1, 1, 1);
        dim3 block_size = dim3(blockSize, 1, 1);
        statistics_reset_d<<<grid_size,block_size>>>(d_fluxC, model_size);
        // for(int i = 0; i < model_size; i++) {
        //     temperature_plot[i] = 0;
        //     fluxC[i] = 0;
        // }
    }
    void output_Kappa() {
        dim3 grid_size = dim3(model_size / blockSize + 1, 1, 1);
        dim3 block_size = dim3(blockSize, 1, 1);
        //statistics_reset_test<<<grid_size,block_size>>>(d_fluxC, state, model_size);
        cudaMemcpy(h_fluxC, d_fluxC, sizeof(float) * model_size, cudaMemcpyDeviceToHost);
        float sum_flux = 0.0;
        int count0 = 0;
        for(int i = n2_L + 1; i < n2_R; i++) {
            sum_flux += h_fluxC[i];
            count0++;
        }
        const float ave_flux = sum_flux / count0 / count_flux;
        const float ave_kappa = ave_flux / (temp_h - temp_l) * middle_size;
        cerr << stepCount << "," << sum_flux << "," << ave_kappa << "," << count_flux << ",";

    }

    // void output_Temperature() {
    //     for(int i = 0; i < model_size; i++) {
    //         cerr << i << "," << temperature_plot[i] / count_temp << endl;
    //     }
    // }

};

int main(void) {
    FPUT_Lattice_1D model = FPUT_Lattice_1D();
    model.settingSize(20, 262144);
    model.settingStep(1000, 10000, 5000000000);
    std::chrono::system_clock::time_point  start, end; // 型は auto で可
    start = std::chrono::system_clock::now(); // 計測開始時間
 

    for(int i = 0; i < 5000000000/statistics_interval;i++) {
        if(i == 10000)
            model.statistics_reset();
        model.step();
        if((i + 1) % (50000/statistics_interval) == 0) {
            model.showProcessing();
            end = std::chrono::system_clock::now();  // 計測終了時間
            float elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
            model.output_Kappa();
            cerr << elapsed/1000.0/60.0/60.0 << "[h]" << "," << elapsed/1000.0/60.0/60.0/i * 5000000000/statistics_interval << "[h]" << endl;
        }
    }
    //model.output_Temperature();
}