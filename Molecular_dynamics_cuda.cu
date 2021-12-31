#include <iostream>
#include <random>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <cufft.h>
#include <cassert>
#define blockSize (64)
//#define mu1 (1)
#define mu1_bath (1)
#define mu0 (0)
#define beta0 (1)
//#define dt (0.05)              // 時間刻み幅
#define dt_1 (1.0 / dt)
#define dt2 (dt * dt)
#define gamma0 (0.2)          // Langevin熱浴のgamma係数
#define gamma_t (gamma0 * dt)   
#define PI (3.1415926535897932384626)
#ifdef _WIN32
    #define SLASH "\\"
#else
    #define SLASH "/"
#endif
std::string outputDir = "data" SLASH + std::to_string(ID) + SLASH "N=" + std::to_string(SIZE) + SLASH;

using namespace std;

__global__ 
void setCurand(unsigned long long seed, curandState *state){
    unsigned int i_global = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, i_global, 0, &state[i_global]);
}
__global__
void qinit(double *q1, double *q0, const int size_)
{
    const int thread_id = threadIdx.x+blockDim.x*blockIdx.x;
    const int i = thread_id;
    if(i < size_) {
        q0[i] = q1[i] = 0.0;
    }
}
__global__
void statistics_reset_d(double* fluxC, double* temperature_plot, const int model_size) {
    const int thread_id = threadIdx.x+blockDim.x*blockIdx.x;
    const int i = thread_id + 1;
    if(i < model_size) {
        fluxC[i] = 0.0;
        temperature_plot[i] = 0.0;
    }
        
}

__global__
void Update(double* q2, const double* q1, const double *q0,curandState *state,  double* ct_c, double *fluxC,double *temperature_plot, const int n1_L, const int n1_R, const int n2_L, const int n2_R, const int n3_L, const int n3_R, const bool is_heat)
{
    const int thread_id = threadIdx.x+blockDim.x*blockIdx.x;
    const int i = thread_id + 1;
    const int is_middleL = (n2_L < i && i <= n2_R);
    const int is_middleR = (n2_L <= i && i < n2_R);
    
    const double muL = is_middleL * mu1 + !(is_middleL) * mu1_bath;
    const double muR = is_middleR * mu1 + !(is_middleR) * mu1_bath;
    double dq2;
    if(i <= n3_R) {
        const double f_R = muR * (q1[i + 1] - q1[i]);
        const double f_L = muL * (q1[i] - q1[i - 1]);
        const double l_f = f_R - f_L;

        const double nf_R = (is_middleR) * beta0 * (q1[i + 1] - q1[i]) * (q1[i + 1] - q1[i]) * (q1[i + 1] - q1[i]);
        const double nf_L = (is_middleL) * beta0 * (q1[i - 1] - q1[i]) * (q1[i - 1] - q1[i]) * (q1[i - 1] - q1[i]);
        const double nl_f = nf_L + nf_R;

        const double dq1 = q1[i] - q0[i];
        const double random_f = (is_heat || i <= n1_R || i >= n3_L) ? (-gamma_t * dq1 + ct_c[i] * curand_normal(&state[i])) : 0;

        const double f = l_f + nl_f - mu0 * q1[i];
        dq2 = dq1 + f * dt2 + random_f;  

        q2[i] = q1[i] + dq2;
    }
}

__global__
void Update2(float* p, const int batch_num, const int middle_size, double* q2, const double* q1, float* q1_f, const double *q0, curandState *state, double* ct_c, double *fluxC,double *temperature_plot, const int n1_L, const int n1_R, const int n2_L, const int n2_R, const int n3_L, const int n3_R, const bool is_heat)
{
    const int thread_id = threadIdx.x+blockDim.x*blockIdx.x;
    const int i = thread_id + 1;
    if(i <= n3_R) {
        const int is_middleL = (n2_L < i && i <= n2_R);
        const int is_middleR = (n2_L <= i && i < n2_R);
        
        const double muL = is_middleL * mu1 + !(is_middleL) * mu1_bath;
        const double muR = is_middleR * mu1 + !(is_middleR) * mu1_bath;

        const double f_R = muR * (q1[i + 1] - q1[i]);
        const double f_L = muL * (q1[i] - q1[i - 1]);
        const double l_f = f_R - f_L;

        const double nf_R = (is_middleR) * beta0 * (q1[i + 1] - q1[i]) * (q1[i + 1] - q1[i]) * (q1[i + 1] - q1[i]);
        const double nf_L = (is_middleL) * beta0 * (q1[i - 1] - q1[i]) * (q1[i - 1] - q1[i]) * (q1[i - 1] - q1[i]);
        const double nl_f = nf_L + nf_R;

        const double dq1 = q1[i] - q0[i];
        const double random_f = (is_heat || i <= n1_R || i >= n3_L) ? (-gamma_t * dq1 + ct_c[i] * curand_normal(&state[i])) : 0;

        const double f = l_f + nl_f - mu0 * q1[i];
        const double dq2 = dq1 + f * dt2 + random_f;  

        q2[i] = q1[i] + dq2;
        q1_f[i] = (float)q2[i];
        // 統計量の計算
        const double p1 = 0.5 * (q2[i] - q0[i]) * dt_1;
        fluxC[i] += p1 * (-f_L + nf_L);
        temperature_plot[i] += p1 * p1;
    }
}
__global__
void calcMomentam(float* p, const double* q2, const double* q0, const int n1, const int middle_size, const int batch_num) {
    const int thread_id = threadIdx.x+blockDim.x*blockIdx.x;
    const int i = thread_id;
    p[i + middle_size * batch_num] = (0.5 * (q2[i + n1] - q0[i + n1]) / dt);
}


__global__
void modalfluxMeasurement(double* Usum, const cufftComplex * d_cplx1, const cufftComplex * d_cplx2, const int middle_size, const int BATCH) {
    const int thread_id = threadIdx.x+blockDim.x*blockIdx.x;
    const int k = thread_id;
    const int k1=middle_size/2+k; // k>0
    const int k2=middle_size/2-k; // k<0
    const float theta=PI*(float)k/(float)middle_size;
    const float omega=2*sqrtf(mu1)*sinf(theta);
    if(k2 >= 0) {
        for(int i = 0; i < BATCH; i++) {
            const int idx = k + (middle_size / 2 + 1) * i;
            const float ad1= omega*d_cplx1[idx].x - d_cplx2[idx].y;
            const float bc1=-omega*d_cplx1[idx].y - d_cplx2[idx].x;
            const float ad2= omega*d_cplx1[idx].x + d_cplx2[idx].y;
            const float bc2= omega*d_cplx1[idx].y - d_cplx2[idx].x;
            Usum[k1]+=0.5*ad1*ad1+0.5*bc1*bc1;
            Usum[k2]+=0.5*ad2*ad2+0.5*bc2*bc2;
        }
    }
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

    const double temp_h = 1.2;              // 高温熱浴の温度
    const double temp_l = 0.8;              // 低温熱浴の温度

    const double c_h = sqrt(2.0*gamma0*temp_h*dt);  // 高温側の分散
    const double c_l = sqrt(2.0*gamma0*temp_l*dt);  // 低温側の分散
    
    const double ct_h = c_h * dt;           
	const double ct_l = c_l * dt;        
    double* h_ct_c;           // 中央部分の（初期）温度  
    double* d_ct_c;           // 中央部分の（初期）温度    

    double* q0;                         // 2ステップ前の粒子位置
    double* q1;                         // 1ステップ前の粒子位置
    float* q1_f;                         // 1ステップ前の粒子位置
    double* q2;                         // 現在の粒子位置    
    double* h_fluxC;
    double* h_temperature_plot;
    double* d_fluxC;
    double* d_temperature_plot;
    double* d_Usum, *h_Usum;
    float* d_real1, *d_real2;
    cufftComplex *d_cplx1, *d_cplx2;
    long long int interval = 32;

    long long int count_flux = 0;
    long long int count_modal_flux = 0;
    long long int count_modal_flux_temp = 0;
    long long int count_modal_flux_actu = 0;
    long long int count_temp = 0;
    long long int stepCount = 0;

    int BATCH;
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
        BATCH = min(1024, (1 << 26) / model_size);
        cerr << n1_L << "," << n1_R << "," << n2_L << "," << n2_R << "," << n3_L << "," << n3_R <<"," << model_size <<  endl;

        h_ct_c = (double *)malloc(sizeof(double)* (model_size + 256));
        h_fluxC = (double *)malloc(sizeof(double) * (model_size + 256));
        h_temperature_plot = (double *)malloc(sizeof(double) * (model_size + 256));
        h_Usum = (double *)malloc(sizeof(double) * (model_size + 256));

        cudaMalloc(&d_real1, (model_size + 256) * sizeof(float) * BATCH);
        cudaMalloc(&d_real2, (model_size + 256) * sizeof(float) * BATCH);
        cudaMalloc((void**)&d_cplx1, sizeof(cufftComplex)*(model_size + 256)*BATCH);
        cudaMalloc((void**)&d_cplx2, sizeof(cufftComplex)*(model_size + 256)*BATCH);

        cudaMalloc(&d_ct_c, sizeof(double) * (model_size + 256));
        cudaMalloc(&d_fluxC, sizeof(double) * (model_size + 256));
        cudaMalloc(&d_temperature_plot, sizeof(double) * model_size);
        cudaMalloc(&d_Usum, sizeof(double) * (model_size + 256));

        cudaMalloc(&q0, sizeof(double) * (model_size + 256));
        cudaMalloc(&q1, sizeof(double) * (model_size + 256));
        cudaMalloc(&q1_f, sizeof(float) * (model_size + 256));
        cudaMalloc(&q2, sizeof(double) * (model_size + 256));
        
        cudaMalloc(&state,  (model_size + 256) * sizeof(curandState));
        for(int i = n1_L; i <= n1_R; i++) {
            h_ct_c[i] = ct_h;
        }
        std::string inputDir = "data" SLASH + std::to_string(ID) + SLASH "N=" + std::to_string(SIZE/2) + SLASH;
        std::string inputFile = inputDir + "Temperature_100.txt";
        std::ifstream inputTemperature(inputFile);
        if (inputTemperature.fail()) {
            for(int i = n2_L; i <= n2_R; i++) {
                const double temp_c = temp_h - (temp_h - temp_l) * (i - n2_L)/(n2_R - n2_L + 1);
                const double c_c = sqrt(2.0*gamma0*temp_c*dt); 
                h_ct_c[i] = c_c * dt;
            }
        } else {
            std::string str, s;
            while (getline(inputTemperature, str)) {
                std::vector<std::string> v; 
                std::stringstream ss{str};             
                while ( getline(ss, s, ' ') ){    
                    v.push_back(s);
                }
                if(v.size() < 2) continue;
                const int i = stoi(v[0]);
                const double temp_ic = stod(v[1]);
                if(i * 2 >= n2_L && i * 2 <= n2_R) {
                    h_ct_c[i * 2] = sqrt(2.0 * gamma0 * temp_ic * dt) * dt;
                    h_ct_c[i * 2 + 1] = sqrt(2.0 * gamma0 * temp_ic * dt) * dt;
                }
            }
            std::cerr << "previous_ok" << std::endl;
        }
        for(int i = n3_L; i <= n3_R; i++) {
            h_ct_c[i] = ct_l;
        }
        cudaMemcpy(d_ct_c, h_ct_c, sizeof(double) * model_size, cudaMemcpyHostToDevice);
        dim3 grid_size = dim3(model_size / blockSize + 1, 1, 1);
        dim3 block_size = dim3(blockSize, 1, 1);
        setCurand<<<grid_size,block_size>>>(1, state);
        qinit<<<grid_size,block_size>>>(q0, q1, model_size);
    }
    // 進行度の出力
    void showProcessing() {
        double *h_q0L = (double*) malloc(sizeof(double));
        double *h_q0M = (double*) malloc(sizeof(double));
        double *h_q0R = (double*) malloc(sizeof(double));
        cudaMemcpy(h_q0L, q0 + (n1_L + n1_R) / 2, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_q0M, q0 + middle_size / 2, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_q0R, q0 + (n3_L + n3_R) / 2, sizeof(double), cudaMemcpyDeviceToHost);
        std::cerr << stepCount << "," << *h_q0L << "," << *h_q0M << "," << *h_q0R << "," << calc_Kappa() << ",";
        assert(*h_q0M < 1e+10);

        free(h_q0L); free(h_q0M); free(h_q0R);
    }

    void step() {
        dim3 grid_size = dim3(model_size / blockSize + 1, 1, 1);
        dim3 block_size = dim3(blockSize, 1, 1);
        if(stepCount % interval == 0 && stepCount >= initialStateSteps) {
            //void Update2(float *p, int batch_num, int middle_size, double *q2, const double *q1, float *q1_f, const double *q0, curandState *state, double *ct_c, double *fluxC, double *temperature_plot, int n1_L, int n1_R, int n2_L, int n2_R, int n3_L, int n3_R, bool is_heat)

            Update2<<<grid_size,block_size>>>(d_real2, count_modal_flux % BATCH, middle_size,  q2, q1,q1_f, q0, state,d_ct_c, d_fluxC, d_temperature_plot, n1_L, n1_R, n2_L, n2_R, n3_L, n3_R, stepCount < initialHeatSteps);
            cudaMemcpy(d_real1 + middle_size * (count_modal_flux % BATCH), q1_f + n2_L, sizeof(float) * middle_size , cudaMemcpyDeviceToDevice);
            calcMomentam<<<grid_size, block_size>>>(d_real2, q2, q0, n2_L, middle_size, (count_modal_flux % BATCH));
            grid_size = dim3(middle_size/blockSize, 1, 1);
            block_size = dim3(blockSize, 1, 1);
            //cerr << stepCount << "," << count_modal_flux << "," << BATCH << endl;
            if((count_modal_flux + 1) % BATCH == 0) {
                cufftHandle plan_f;
                int n[] = {middle_size};
                int istride = 1, ostride = 1;           // --- Distance between two successive input/output elements
                int idist = middle_size, odist = (middle_size / 2 + 1);      // --- Distance between batches
                int inembed[] = { 0 };                  // --- Input size with pitch (ignored for 1D transforms)
                int onembed[] = { 0 };                  // --- Output size with pitch (ignored for 1D transforms)
                cufftPlanMany(&plan_f, 1, n, 
                    inembed, istride, idist,
                    onembed, ostride, odist, CUFFT_R2C, BATCH); // Real to Complex (forward)
                cufftExecR2C(plan_f, d_real1, d_cplx1);
                cufftExecR2C(plan_f, d_real2, d_cplx2);
                grid_size = dim3(model_size/blockSize, 1, 1);
                block_size = dim3(blockSize/2, 1, 1);
                modalfluxMeasurement<<<grid_size, block_size>>>(d_Usum, d_cplx1, d_cplx2, middle_size, BATCH);
                cufftDestroy(plan_f);
                count_modal_flux_actu += count_modal_flux_temp;
                count_modal_flux_temp = 0;
                
            }
            count_flux++;
            count_temp++;
            count_modal_flux++;
            count_modal_flux_temp++;
            swap(q0, q1);
            swap(q1, q2);
        } else {
            Update<<<grid_size,block_size>>>(q2, q1, q0, state,d_ct_c, d_fluxC, d_temperature_plot, n1_L, n1_R, n2_L, n2_R, n3_L, n3_R, stepCount < initialHeatSteps);
            swap(q0, q1);
            swap(q1, q2);
        }
        
        
        stepCount += 1;
    }
    void statistics_reset() {
        count_flux = 0;
        count_temp = 0;
        dim3 grid_size = dim3(model_size / blockSize + 1, 1, 1);
        dim3 block_size = dim3(blockSize, 1, 1);
        statistics_reset_d<<<grid_size,block_size>>>(d_fluxC, d_temperature_plot, model_size);
    }
    double calc_Kappa() {
        dim3 grid_size = dim3(model_size / blockSize + 1, 1, 1);
        dim3 block_size = dim3(blockSize, 1, 1);
        //statistics_reset_test<<<grid_size,block_size>>>(d_fluxC, state, model_size);
        cudaMemcpy(h_fluxC, d_fluxC, sizeof(double) * model_size, cudaMemcpyDeviceToHost);
        double sum_flux = 0.0;
        int count0 = 0;
        for(int i = n2_L + 1; i < n2_R; i++) {
            sum_flux += h_fluxC[i];
            count0++;
        }
        const double ave_flux = sum_flux / count0 / count_flux;
        const double ave_kappa = ave_flux / (temp_h - temp_l) * middle_size;
        return ave_kappa;
    }
    void output_setting() {
        std::string FileNameSetting = outputDir + "Setting.txt";
        std::ofstream settingOutputFile(FileNameSetting);
        settingOutputFile << "n," << middle_size << endl;
        settingOutputFile << "HeatBath_size," << HeatBath_size << endl;
        settingOutputFile << "allSteps," << allSteps << endl;
        settingOutputFile << "initialStateSteps," << initialStateSteps << endl;
        settingOutputFile << "initialHeatSteps," << initialHeatSteps << endl;
        settingOutputFile << "dt," << dt << endl;
        settingOutputFile << "mu0," << mu0 << endl;
        settingOutputFile << "mu1," << mu1 << endl;
        settingOutputFile << "beta0," << beta0 << endl;
    }

    void output_Kappa() {
        std::string FileNameKappa = outputDir + "kappa.txt";
        std::ofstream kappaOutputFile(FileNameKappa, std::ios::app);
        kappaOutputFile << stepCount << "," <<  std::fixed << std::setprecision(12) << calc_Kappa() << "," << count_flux << endl;
    }

    void output_Temperature() {
        const int num = (double)stepCount/(double)allSteps * 100.0 + 0.01;     
        std::ostringstream sout;
        sout << "_" << std::setfill('0') << std::setw(3) << num;
        std::string num0 = sout.str(); 
        std::string FileNameTemp = outputDir + "Temperature" + num0 + ".txt";
        std::ofstream temperatureOutputFile(FileNameTemp);

        cudaMemcpy(h_temperature_plot, d_temperature_plot, sizeof(double) * model_size, cudaMemcpyDeviceToHost);
        for(int i = 0; i < model_size; i++) {
            temperatureOutputFile << i << "," << std::fixed << std::setprecision(12) << h_temperature_plot[i] / count_temp << "," << count_temp << endl;
        }
    }
    void output_modalFlux() {
        const int num = (double)stepCount/(double)allSteps * 100.0 + 0.01;     
        std::ostringstream sout;
        sout << "_" << std::setfill('0') << std::setw(3) << num;
        std::string num0 = sout.str(); 
        std::string FileNameModalFlux = outputDir + "modalFlux" + num0 + ".txt";
        std::ofstream modalFluxOutputFile(FileNameModalFlux);

        cudaMemcpy(h_Usum, d_Usum, middle_size * sizeof(double), cudaMemcpyDeviceToHost);
        if(count_modal_flux_actu == 0) count_modal_flux_actu = 1;
        for(int km=0; km<middle_size/2; km++){
            const int km1=middle_size/2+km; // k>0
            const int km2=middle_size/2-km; // k<0
            const double vg1=cos(PI*(double)km/(double)middle_size);
            const double vg2=-vg1;
            const double mode_flux1=vg1*h_Usum[km1]/(double)count_modal_flux_actu/(double)middle_size;
            const double mode_flux2=vg2*h_Usum[km2]/(double)count_modal_flux_actu/(double)middle_size;
            const double mode_flux = mode_flux1+mode_flux2;
            modalFluxOutputFile << km << "," << std::fixed << std::setprecision(8) << (double)km/(double)middle_size << "," << mode_flux << "," << mode_flux1 << "," << mode_flux2 << "," << count_modal_flux_actu << endl;
        }
    }

};

int main(void) {
    FPUT_Lattice_1D model = FPUT_Lattice_1D();

    const long long int output_file_interval = allSteps / 20;
    const long long int output_cerr_interval = allSteps / 1000;
    const long long int output_cerr_interval2 = allSteps / 50000;
    model.settingSize(20, SIZE);
    std::chrono::system_clock::time_point  start, end; // 型は auto で可
    start = std::chrono::system_clock::now(); // 計測開始時間
    model.output_setting();
    for(int i = 0; i < allSteps;i++) {
        if(i == initialStateSteps)
            model.statistics_reset();
        model.step();
        if((i + 1) % output_cerr_interval == 0 || (i < initialHeatSteps && (i + 1) % output_cerr_interval2 == 0)) {
            model.showProcessing();
            end = std::chrono::system_clock::now();  // 計測終了時間
            double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
            if(i > initialStateSteps)
                model.output_Kappa();
            cerr << elapsed/1000.0/60.0/60.0 << "[h]" << "," << elapsed/1000.0/60.0/60.0/i * allSteps << "[h]" << endl;
        }
        if((i + 1) % output_file_interval == 0 && i > initialStateSteps) {
            model.output_Temperature();
            model.output_modalFlux();
        }
    }
    
}