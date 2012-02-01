/*
 * ActivactionFunctionTanh.cu
 *
 *  Created on: Jan 30, 2012
 *      Author: wchan
 */

/*
 * Note: This file is needed because the nvcc compiler doesn't support C++0x
 *
 * When the nvcc compiler support comes, this can be integrated back in the .hpp file
 */

#include "ActivationFunctionTanh.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

struct tanh_func {
  __device__
  double operator()(double x) {
    return tanh(x);
  }
};

struct dtanh_func {
  __device__
  double operator()(double y) {
    return 1.0 - y * y;
  }
};

void ActivationFunctionTanh::f_cuda(double* x, size_t size) {
  thrust::device_ptr<double> ptr(x);

  thrust::transform(ptr, ptr + size, ptr, tanh_func());
}

void ActivationFunctionTanh::d_cuda(double* y, size_t size, double* d) {
  thrust::device_ptr<double> ptr0(y);
  thrust::device_ptr<double> ptr1(d);

  thrust::transform(ptr0, ptr0 + size, ptr1, dtanh_func());
}

