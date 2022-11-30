/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

 /* Example showing the use of CUFFT for fast 1D-convolution using FFT. */

 // includes, system
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <device_launch_parameters.h>
#include "Common/helper_cuda.h"
#include "Common/helper_functions.h"
#include "Common/helper_timer.h"

int idxNum = 4;
int startIdx[] = { 0, 1, 2, 4 };

// Complex data type
typedef float2 Complex;
static __device__ __host__ inline Complex ComplexAdd(Complex, Complex);
static __device__ __host__ inline Complex ComplexScale(Complex, float);
static __device__ __host__ inline Complex ComplexMul(Complex, Complex);
static __device__ __host__ inline float ComplexMag(Complex);
static __global__ void ComplexPointwiseMulAndScale(Complex*, const Complex*, int, float);
static __global__ void ComplexArrayScale(Complex* a, int size, float scale);
static __global__ void SegmentFun(Complex* a, int size);
static __global__ void ArrayFFT(
    Complex* hostArray,
    Complex* deviceArray,
    int arraySize,
    int segmentSize,
    cufftHandle& plan);
static __global__ void ArrayIFFT(Complex* a, int size);
static __global__ void ArrayFun(Complex* a, int size);

// Filtering functions
void Convolve(const Complex*, int, const Complex*, int, Complex*);

// Padding functions
int PadData(const Complex*, Complex**, int, const Complex*, Complex**, int);

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);

// The filter size is assumed to be a number smaller than the signal size
#define SIGNAL_SIZE 50
#define FILTER_KERNEL_SIZE 11

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) { runTest(argc, argv); }

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
#if 1
void readFile(Complex* buf, int numberNum)
{
    FILE* fp;
    fopen_s(&fp, "D:/qyx-dev/MatlabTest/complex_data.bin", "rb");
    fread(buf, sizeof(float), numberNum, fp);
    fclose(fp);
}

void printData(Complex* buf, int numberNum)
{
    printf("printData\n");
    for (int i = 0; i < numberNum; ++i)
    {
        printf("complex %f %f\n", buf[i].x, buf[i].y);
    }
}

void runTest(int argc, char** argv)
{
    findCudaDevice(argc, (const char**)argv);

    int size = 32;
    int mem_size = sizeof(Complex) * size;
    Complex* h_filter_kernel =
        reinterpret_cast<Complex*>(malloc(mem_size));
    readFile(h_filter_kernel, size * 2);
    //printData(h_filter_kernel, size);

    StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    Complex* d_signal;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_signal), mem_size));
    checkCudaErrors(cudaMemcpy(d_signal, h_filter_kernel, mem_size, cudaMemcpyHostToDevice));
    cufftHandle plan;
    checkCudaErrors(cufftPlan1d(&plan, size, CUFFT_C2C, 1));

    //sdkStartTimer(&timer);

    checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(d_signal), reinterpret_cast<cufftComplex*>(d_signal), CUFFT_FORWARD));
    checkCudaErrors(cudaMemcpy(h_filter_kernel, d_signal, mem_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(d_signal, h_filter_kernel, mem_size, cudaMemcpyHostToDevice));
    SegmentFun << <32, 256 >> > (d_signal, size);
    checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(d_signal), reinterpret_cast<cufftComplex*>(d_signal), CUFFT_INVERSE));
    ComplexArrayScale << <32, 256 >> > (d_signal, size, 0.03125);// 1/32

    checkCudaErrors(cudaMemcpy(h_filter_kernel, d_signal, mem_size, cudaMemcpyDeviceToHost));
    sdkStopTimer(&timer);
    printf("time spent by CPU in CUDA calls: %.2fms\n", sdkGetTimerValue(&timer));

    //printData(h_filter_kernel, size);
}
#else
void runTest(int argc, char** argv) {
    printf("[simpleCUFFT] is starting...\n");

    findCudaDevice(argc, (const char**)argv);

    // Allocate host memory for the signal
    Complex* h_signal =
        reinterpret_cast<Complex*>(malloc(sizeof(Complex) * SIGNAL_SIZE));

    // Initialize the memory for the signal
    for (unsigned int i = 0; i < SIGNAL_SIZE; ++i) {
        h_signal[i].x = rand() / static_cast<float>(RAND_MAX);
        h_signal[i].y = 0;
    }

    // Allocate host memory for the filter
    Complex* h_filter_kernel =
        reinterpret_cast<Complex*>(malloc(sizeof(Complex) * FILTER_KERNEL_SIZE));

    // Initialize the memory for the filter
    for (unsigned int i = 0; i < FILTER_KERNEL_SIZE; ++i) {
        h_filter_kernel[i].x = rand() / static_cast<float>(RAND_MAX);
        h_filter_kernel[i].y = 0;
    }
    StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
    
    // Pad signal and filter kernel
    Complex* h_padded_signal;
    Complex* h_padded_filter_kernel;
    int new_size =
        PadData(h_signal, &h_padded_signal, SIGNAL_SIZE, h_filter_kernel,
            &h_padded_filter_kernel, FILTER_KERNEL_SIZE);
    int mem_size = sizeof(Complex) * new_size;

    // Allocate device memory for signal
    Complex* d_signal;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_signal), mem_size));
    // Copy host memory to device
    checkCudaErrors(
        cudaMemcpy(d_signal, h_padded_signal, mem_size, cudaMemcpyHostToDevice));

    // Allocate device memory for filter kernel
    Complex* d_filter_kernel;
    checkCudaErrors(
        cudaMalloc(reinterpret_cast<void**>(&d_filter_kernel), mem_size));

    // Copy host memory to device
    checkCudaErrors(cudaMemcpy(d_filter_kernel, h_padded_filter_kernel, mem_size,
        cudaMemcpyHostToDevice));

    // CUFFT plan simple API
    cufftHandle plan;
    checkCudaErrors(cufftPlan1d(&plan, new_size, CUFFT_C2C, 1));

    // CUFFT plan advanced API
    cufftHandle plan_adv;
    size_t workSize;
    long long int new_size_long = new_size;

    checkCudaErrors(cufftCreate(&plan_adv));
    checkCudaErrors(cufftXtMakePlanMany(plan_adv, 1, &new_size_long, NULL, 1, 1,
        CUDA_C_32F, NULL, 1, 1, CUDA_C_32F, 1,
        &workSize, CUDA_C_32F));
    printf("Temporary buffer size %li bytes\n", workSize);

    // Transform signal and kernel
    printf("Transforming signal cufftExecC2C\n");
    sdkStartTimer(&timer);
    checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(d_signal),
        reinterpret_cast<cufftComplex*>(d_signal),
        CUFFT_FORWARD));
    checkCudaErrors(cufftExecC2C(
        plan_adv, reinterpret_cast<cufftComplex*>(d_filter_kernel),
        reinterpret_cast<cufftComplex*>(d_filter_kernel), CUFFT_FORWARD));
    sdkStopTimer(&timer);
    printf("time spent by CPU in CUDA calls: %.2f\n", sdkGetTimerValue(&timer));
    // Multiply the coefficients together and normalize the result
    printf("Launching ComplexPointwiseMulAndScale<<< >>>\n");
    sdkStartTimer(&timer);
    ComplexPointwiseMulAndScale << <32, 256 >> > (d_signal, d_filter_kernel, new_size,
        1.0f / new_size);
    sdkStopTimer(&timer);
    printf("time spent by CPU in CUDA calls: %.2f\n", sdkGetTimerValue(&timer));
    // Check if kernel execution generated and error
    getLastCudaError("Kernel execution failed [ ComplexPointwiseMulAndScale ]");

    // Transform signal back
    printf("Transforming signal back cufftExecC2C\n");
    checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(d_signal),
        reinterpret_cast<cufftComplex*>(d_signal),
        CUFFT_INVERSE));

    // Copy device memory to host
    Complex* h_convolved_signal = h_padded_signal;
    checkCudaErrors(cudaMemcpy(h_convolved_signal, d_signal, mem_size,
        cudaMemcpyDeviceToHost));

    // Allocate host memory for the convolution result
    Complex* h_convolved_signal_ref =
        reinterpret_cast<Complex*>(malloc(sizeof(Complex) * SIGNAL_SIZE));

    // Convolve on the host
    Convolve(h_signal, SIGNAL_SIZE, h_filter_kernel, FILTER_KERNEL_SIZE,
        h_convolved_signal_ref);

    // check result
    bool bTestResult = sdkCompareL2fe(
        reinterpret_cast<float*>(h_convolved_signal_ref),
        reinterpret_cast<float*>(h_convolved_signal), 2 * SIGNAL_SIZE, 1e-5f);

    // Destroy CUFFT context
    checkCudaErrors(cufftDestroy(plan));
    checkCudaErrors(cufftDestroy(plan_adv));

    // cleanup memory
    free(h_signal);
    free(h_filter_kernel);
    free(h_padded_signal);
    free(h_padded_filter_kernel);
    free(h_convolved_signal_ref);
    checkCudaErrors(cudaFree(d_signal));
    checkCudaErrors(cudaFree(d_filter_kernel));

    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
#endif
// Pad data
int PadData(const Complex* signal, Complex** padded_signal, int signal_size,
    const Complex* filter_kernel, Complex** padded_filter_kernel,
    int filter_kernel_size) {
    int minRadius = filter_kernel_size / 2;
    int maxRadius = filter_kernel_size - minRadius;
    int new_size = signal_size + maxRadius;

    // Pad signal
    Complex* new_data =
        reinterpret_cast<Complex*>(malloc(sizeof(Complex) * new_size));
    memcpy(new_data + 0, signal, signal_size * sizeof(Complex));
    memset(new_data + signal_size, 0, (new_size - signal_size) * sizeof(Complex));
    *padded_signal = new_data;

    // Pad filter
    new_data = reinterpret_cast<Complex*>(malloc(sizeof(Complex) * new_size));
    memcpy(new_data + 0, filter_kernel + minRadius, maxRadius * sizeof(Complex));
    memset(new_data + maxRadius, 0,
        (new_size - filter_kernel_size) * sizeof(Complex));
    memcpy(new_data + new_size - minRadius, filter_kernel,
        minRadius * sizeof(Complex));
    *padded_filter_kernel = new_data;

    return new_size;
}

////////////////////////////////////////////////////////////////////////////////
// Filtering operations
////////////////////////////////////////////////////////////////////////////////

// Computes convolution on the host
void Convolve(const Complex* signal, int signal_size,
    const Complex* filter_kernel, int filter_kernel_size,
    Complex* filtered_signal) {
    int minRadius = filter_kernel_size / 2;
    int maxRadius = filter_kernel_size - minRadius;

    // Loop over output element indices
    for (int i = 0; i < signal_size; ++i) {
        filtered_signal[i].x = filtered_signal[i].y = 0;

        // Loop over convolution indices
        for (int j = -maxRadius + 1; j <= minRadius; ++j) {
            int k = i + j;

            if (k >= 0 && k < signal_size) {
                filtered_signal[i] =
                    ComplexAdd(filtered_signal[i],
                        ComplexMul(signal[k], filter_kernel[minRadius - j]));
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex addition
static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b) {
    Complex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

// Complex scale
static __device__ __host__ inline Complex ComplexScale(Complex a, float s) {
    Complex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

// Complex multiplication
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b) {
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

// Complex mag
static __device__ __host__ inline float ComplexMag(Complex a)
{
    return sqrtf(a.x * a.x + a.y * a.y);
}

static __global__ void ComplexArrayScale(Complex* a, int size, float scale)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < size; i += numThreads)
    {
        a[i] = ComplexScale(a[i], scale);
    }
}

static __global__ void ArrayFFT(
    Complex* hostArray,
    Complex* deviceArray,
    int arraySize,
    int segmentSize,
    cufftHandle &plan)
{
    // The length of hostArray is arraySize
    // The length of deviceArray is (idxNum * segmentSize)
    size_t mem_size = segmentSize * sizeof(Complex);
    for (int i = 0; i < idxNum; ++i)
    {
        Complex* host = hostArray + startIdx[i] * segmentSize;
        Complex* device = deviceArray + i * segmentSize;
        checkCudaErrors(cudaMemcpy(device, host, mem_size, cudaMemcpyHostToDevice));
        checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(device), reinterpret_cast<cufftComplex*>(device), CUFFT_FORWARD));
        checkCudaErrors(cudaMemcpy(host, device, mem_size, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(device, host, mem_size, cudaMemcpyHostToDevice));
    }
}

static __global__ void ArrayIFFT(Complex* a, int size)
{
    for (int i = 0; i < idxNum; ++i)
    {
//        SegmentFun(a + startIdx[i], 32);
    }
}

static __global__ void ArrayFun(Complex* a, int size)
{
    for (int i = 0; i < idxNum; ++i)
    {
//        SegmentFun(a + startIdx[i], 32);
    }
}

static __global__ void SegmentFun(Complex* a, int size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < size; i += numThreads)
    {
        float inv = 1.0f / ComplexMag(a[i]);
        a[i] = ComplexScale(a[i], inv);
    }
}

// Complex pointwise multiplication
static __global__ void ComplexPointwiseMulAndScale(Complex* a, const Complex* b, int size, float scale) 
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < size; i += numThreads) {
        a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
    }
}
