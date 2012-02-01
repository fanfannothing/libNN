/*
 * BackpropagationCUDA.cu
 *
 *  Created on: Jan 30, 2012
 *      Author: wchan
 */

/**
 * This file is needed because nvcc doesn't support C++0x yet... we can merge it back in later when nvcc adds support for the C++11 standard
 */

#include <thrust/device_ptr.h>
#include <thrust/transform.h>

void mult(double* x, double* y, size_t count) {
  thrust::device_ptr<double> x_ptr(x);
  thrust::device_ptr<double> y_ptr(y);

  thrust::transform(x_ptr, x_ptr + count, y_ptr, x_ptr, thrust::multiplies<double>());
}
