#include <iostream>
#include <random>
#include <cmath>
#include <chrono>
#include <stdlib.h>
#include <stdio.h>

using namespace std;
mt19937_64 mt;                          // 乱数器
std::normal_distribution<> norm(0, 1.0);// 標準正規分布(平均0,標準偏差1)
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

    const double dt = 0.05;                 // 時間刻み幅
    const double dt_1 = 1.0 / dt;
    const double dt2 = dt * dt;

    const double mu0 = 0.0;                 // オンサイトポテンシャル (自分の粒子)
    const double mu1 = 1.0;                 // オンサイトポテンシャル (カップリング)
    const double mu1_bath = 1.0;            // 熱浴オンサイトポテンシャル
    const double beta0 = 1.0;

    const double temp_h = 1.2;              // 高温熱浴の温度
    const double temp_l = 0.8;              // 低温熱浴の温度

    const double gamma0 = 0.2;          // Langevin熱浴のgamma係数
    const double c_h = sqrt(2.0*gamma0*temp_h*dt);  // 高温側の分散
    const double c_l = sqrt(2.0*gamma0*temp_l*dt);  // 低温側の分散
    
    const double gamma_t = gamma0 * dt;   
    const double ct_h = c_h * dt;           
	const double ct_l = c_l * dt;        
    double* ct_c;           // 中央部分の（初期）温度   

    double* q0;                         // 2ステップ前の粒子位置
    double* q1;                         // 1ステップ前の粒子位置
    double* q2;                         // 現在の粒子位置    
    double* fluxC;
    double* temperature_plot;

    int statistics_interval = 16;
    long long int count_flux = 0;
    long long int count_temp = 0;
    long long int stepCount = 0;


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
        ct_c = (double*) malloc(sizeof(double) * model_size);
        q0 = (double*) malloc(sizeof(double) * model_size);
        q1 = (double*) malloc(sizeof(double) * model_size);
        q2 = (double*) malloc(sizeof(double) * model_size);
        fluxC = (double*) malloc(sizeof(double) * model_size);
        temperature_plot = (double*) malloc(sizeof(double) * model_size);
        for(int i = n1_L; i <= n1_R; i++) {
            ct_c[i] = ct_h;
        }
        for(int i = n2_L; i <= n2_R; i++) {
            const double temp_c = temp_h - (temp_h - temp_l) * (i - n2_L)/(n2_R - n2_L + 1);
            const double c_c = sqrt(2.0*gamma0*temp_c*dt); 
            ct_c[i] = c_c * dt;
        }
        for(int i = n3_L; i <= n3_R; i++) {
            ct_c[i] = ct_l;
        }
    }
    void settingStep(const int HeatSimulation_, const int initialStateStep_, const int Step_) {
        HeatSimulation = HeatSimulation_;
        initialStateStep = initialStateStep_;
        Step = Step_;
    }
    void step() {
        for (int k = 0; k < statistics_interval; k++) {
            for(int i = n1_L; i <= n3_R; i++) {
                // 次の粒子状態の計算
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
                const double random_f = (stepCount < HeatSimulation || i <= n1_R || i >= n3_L) ? (-gamma_t * dq1 + ct_c[i] * norm(mt)) : 0;
                
                const double f = l_f + nl_f - mu0 * q1[i] + random_f;
                const double dq2 = dq1 + f * dt2 + random_f;  

                q2[i] = q1[i] + dq2;

                // 統計量の計算
                const double p = 0.5 * (q2[i] - q0[i]) * dt_1;
                fluxC[i] += p * (-f_L + nf_L);
                temperature_plot[i] += p * p;
            }
            count_flux++;
            count_temp++;
            std::swap(q0, q1);      //q0 <- q1
            std::swap(q1, q2);      //q1 <- q2
            stepCount++;
        }
    }
    void statistics_reset() {
        count_flux = 0;
        count_temp = 0;
        for(int i = 0; i < model_size; i++) {
            temperature_plot[i] = 0;
            fluxC[i] = 0;
        }
    }
    void output_Kappa() {
        double sum_flux = 0.0;
        int count0 = 0;
        for(int i = n2_L + 1; i < n2_R; i++) {
            sum_flux += fluxC[i];
            count0++;
        }
        const double ave_flux = sum_flux / count0 / count_flux;
        const double ave_kappa = ave_flux / (temp_h - temp_l) * middle_size;
        cerr << stepCount << "," << sum_flux << "," << ave_kappa << ",";

    }

    void output_Temperature() {
        for(int i = 0; i < model_size; i++) {
            cerr << i << "," << temperature_plot[i] / count_temp << endl;
        }
    }

};

int main(void) {
    FPUT_Lattice_1D model = FPUT_Lattice_1D();
    model.settingSize(20, 256);
    model.settingStep(10000, 100000, 1000000000);
    std::chrono::system_clock::time_point  start, end; // 型は auto で可
    start = std::chrono::system_clock::now(); // 計測開始時間
 

    for(int i = 0; i < 1000000000/16;i++) {
        if(i == 100000/16)
            model.statistics_reset();
        model.step();
        if((i + 1) % (1000000/16) == 0) {
            end = std::chrono::system_clock::now();  // 計測終了時間
            double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
            model.output_Kappa();
            cerr << elapsed/1000.0/60.0/60.0 << "[h]" << "," << elapsed/1000.0/60.0/60.0/i * 1000000000/16 << "[h]" << endl;
        }
    }
    model.output_Temperature();
}