#include "cuda_runtime.h" 
#include "device_launch_parameters.h" 
#include <stdio.h> 
#include <iostream>  
#include<time.h> 
using namespace std;

#define N 2500 //размеры СЛАУ, размеры главной диаганали. 
#define M 1024 //количество СЛАУ 

__global__ void addKernel(double* massC0, double* massC1, double* massC2, double* massD, double* devRez, double* P, double* Q, double* zn)
{
    int kk = blockIdx.x * blockDim.x + threadIdx.x;
   //метод прогонки 
    zn[kk] = massC1[kk * N];
    Q[kk * N] = massD[kk * N] / zn[kk];
    P[kk * (N - 1)] = -massC0[kk * (N - 1)] / zn[kk];
    for (int i = 1; i < N - 1; i++) {
        zn[kk] = massC1[kk * N + i] + massC2[kk * (N - 1) + i] * P[kk * (i - 1)];
        P[kk * (N - 1) + i] = -massC0[kk * (N - 1) + i] / zn[kk];
        Q[kk * N + i] = (massD[kk * N + i] - massC2[kk * (N - 1) + i] * Q[kk * (i - 1)]) / zn[kk];
    }
    zn[kk] = massC1[kk * N + (N - 1)] + massC2[kk * (N - 1) + (N - 2)] * P[kk * (N - 1) + (N - 2)];
    Q[kk * N + (N - 1)] = (massD[kk * N + (N - 1)] - massC2[kk * (N - 1) + (N - 2)] * Q[kk * N + (N - 2)]) / zn[kk];
    devRez[kk * N + (N - 1)] = Q[kk * N + (N - 1)];
    for (int i = N - 2; i > -1; i--) {
        devRez[kk * N + i] = P[kk * (N - 1) + i] * devRez[kk * N + (i + 1)] + Q[kk * N + i];
    }
}

/*__global__ void addKernel_shared(double* massC0, double* massC1, double* massC2, double* massD, double* devRez, double* P, double* Q, double* zn)
{
    int kk = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ double sC0[N - 1], double sC2[N - 1], double sC1[N], double sD[N];
    for (int i = 0; i < N - 1; i++) {
        sC0[i] = massC0[kk * (N - 1) + i];
        sC2[i] = massC0[kk * (N - 1) + i];
    }
    for (int i = 0; i < N; i++) {
        sC1[i] = massC1[kk * N + i];
        sD[i] = massD[kk * N + i];
    }

    //метод прогонки 
    zn[kk] = sC1[N];
    Q[kk * N] = sD[N] / zn[kk];
    P[kk * (N - 1)] = -sC0[(N - 1)] / zn[kk];
    for (int i = 1; i < N - 1; i++) {
        zn[kk] = sC1[i] + sC2[i] * P[kk * (i - 1)];
        P[kk * (N - 1) + i] = -sC0[i] / zn[kk];
        Q[kk * N + i] = (sD[i] - sC2[i] * Q[kk * (i - 1)]) / zn[kk];
    }
    zn[kk] = sC1[N - 1] + sC2[(N - 2)] * P[kk * (N - 1) + (N - 2)];
    Q[kk * N + (N - 1)] = (sD[(N - 1)] - massC2[(N - 2)] * Q[kk * N + (N - 2)]) / zn[kk];
    devRez[kk * N + (N - 1)] = Q[kk * N + (N - 1)];
    for (int i = N - 2; i > -1; i--) {
        devRez[kk * N + i] = P[kk * (N - 1) + i] * devRez[kk * N + (i + 1)] + Q[kk * N + i];
    }
}*/

int main()
{
    int numThread = 1024; //количество нитий 
    double* massC0 = new double[M * (N - 1)];
    double* massC1 = new double[M * N];
    double* massC2 = new double[M * (N - 1)];
    double* massD = new double[M * N];
    double* massRez = new double[M * N];
    double* cpu_massRez = new double[M * N];
    double* P = new double[M * (N - 1)];
    double* Q = new double[M * N];
    double* zn = new double[M];
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            massC1[i * N + j] = rand();
            massD[i * N + j] = rand();
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N - 1; j++) {
            massC0[i * (N - 1) + j] = rand();
            massC2[i * (N - 1) + j] = rand();
        }
    }
    unsigned int cpu_start = clock();
    for (int kk = 0; kk < M; kk++) {
        zn[kk] = massC1[kk * N];
        Q[kk * N] = massD[kk * N] / zn[kk];
        P[kk * (N - 1)] = -massC0[kk * (N - 1)] / zn[kk];
        for (int i = 1; i < N - 1; i++) {
            zn[kk] = massC1[kk * N + i] + massC2[kk * (N - 1) + i] * P[kk * (i - 1)];
            P[kk * (N - 1) + i] = -massC0[kk * (N - 1) + i] / zn[kk];
            Q[kk * N + i] = (massD[kk * N + i] - massC2[kk * (N - 1) + i] * Q[kk * (i - 1)]) / zn[kk];
        }
        zn[kk] = massC1[kk * N + (N - 1)] + massC2[kk * (N - 1) + (N - 2)] * P[kk * (N - 1) + (N - 2)];
        Q[kk * N + (N - 1)] = (massD[kk * N + (N - 1)] - massC2[kk * (N - 1) + (N - 2)] * Q[kk * N + (N - 2)]) / zn[kk];
        cpu_massRez[kk * N + (N - 1)] = Q[kk * N + (N - 1)];
        for (int i = N - 2; i > -1; i--) {
            cpu_massRez[kk * N + i] = P[kk * (N - 1) + i] * cpu_massRez[kk * N + (i + 1)] + Q[kk * N + i];
        }
    }
    unsigned int cpu_end = clock();
    printf("cpu time= %.5f seconds\n", (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC);
    FILE* file_cpu;
    file_cpu = fopen("cpu.txt", "w");
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            fprintf(file_cpu, "%.3f  ", cpu_massRez[i * N + j]);
        }
    double* devC0 = 0;
    double* devC1 = 0;
    double* devC2 = 0;
    double* devD = 0;
    double* devRez = 0;
    double* devP = 0;
    double* devQ = 0;
    double* devzn = 0;
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);  //инцелизируем усстройство 
    cudaStatus = cudaMalloc((void**)&devC0, M * (N - 1) * sizeof(double));  //выделяем память на графическом процессоре 
    cudaStatus = cudaMalloc((void**)&devC1, M * N * sizeof(double));
    cudaStatus = cudaMalloc((void**)&devC2, M * (N - 1) * sizeof(double));
    cudaStatus = cudaMalloc((void**)&devD, M * N * sizeof(double));
    cudaStatus = cudaMalloc((void**)&devRez, M * N * sizeof(double));
    cudaStatus = cudaMalloc((void**)&devP, M * (N - 1) * sizeof(double));
    cudaStatus = cudaMalloc((void**)&devQ, M * N * sizeof(double));
    cudaStatus = cudaMalloc((void**)&devzn, M * sizeof(double));
    cudaStatus = cudaMemcpy(devC0, massC0, M * (N - 1) * sizeof(double), cudaMemcpyHostToDevice);  //копируем с хоста на видеокарту 
    cudaStatus = cudaMemcpy(devC1, massC1, M * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(devC2, massC2, M * (N - 1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(devD, massD, M * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(devRez, massRez, M * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(devP, P, M * (N - 1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(devQ, Q, M * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(devzn, zn, M * sizeof(double), cudaMemcpyHostToDevice);
    int numBlock = M / numThread;
    if (numBlock * numThread < M) numBlock++;
    printf("%d\n", numBlock);
    //unsigned int gpu_start = clock(); 
    float gpuTime;
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start, 0);
    addKernel << <numBlock, numThread >> > (devC0, devC1, devC2, devD, devRez, devP, devQ, devzn);  //выполняем алгаритм на видеокарте  <<<количество блоков, количество нитий>>> 
    cudaEventRecord(gpu_stop, 0);
    cudaStatus = cudaDeviceSynchronize();  //функция синхронизации 
    //unsigned int gpu_end = clock(); 
    cudaEventElapsedTime(&gpuTime, gpu_start, gpu_stop);
    printf("gpu time= %.5f seconds\n", gpuTime / 1000);
    cudaStatus = cudaMemcpy(massRez, devRez, M * N * sizeof(double), cudaMemcpyDeviceToHost);   //копируем результат с видеокарты на хост
    FILE* file_gpu;                        //записываем результат в файл 
    file_gpu = fopen("gpu.txt", "w");
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            fprintf(file_gpu, "%.3f  ", massRez[i * N + j]);
        }
    cudaFree(devC0);
    cudaFree(devC1);
    cudaFree(devC2);
    cudaFree(devD);
    cudaFree(devRez);
    cudaFree(devP);
    cudaFree(devQ);
    cudaFree(devzn);
    cudaStatus = cudaDeviceReset();
    free(massC0);
    free(massC1);
    free(massC2);
    free(massD);
    free(massRez);
    free(cpu_massRez);
    free(P);
    free(Q);
    free(zn);
    system("pause");
    return 0;
}
