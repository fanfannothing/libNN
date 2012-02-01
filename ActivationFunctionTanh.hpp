/*
 * ActivationFunctionTanh.h
 *
 *  Created on: Jan 21, 2012
 *      Author: wchan
 */

#ifndef ACTIVATIONFUNCTIONTANH_H_
#define ACTIVATIONFUNCTIONTANH_H_

#include "ActivationFunction.hpp"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <iostream>

class ActivationFunctionTanh : public ActivationFunction {
public:
  ActivationFunctionTanh() {
  }
  virtual ~ActivationFunctionTanh() {
  }
  static double f(double x) {
    // double y = 1.7159 * std::tanh(0.666666667 * x);
    double y = std::tanh(x);
    return y;
  }
  static double d(double x, double y) {
    // return 0.666666667 / 1.7159 * (1.7159 - y) * (1.7159 + y);
    return 1.0 - y * y;
  }
  static double bound(double y) {
    if (y > 0) return 1;
    return -1;
  }

  /* these two puppies need to be implemented in ActivationFunctionTanh.cu due nvcc not supporting C++0x */
  static void f_cuda(double* x, size_t size);
  static void d_cuda(double* y, size_t size, double* d);
};

#endif /* ACTIVATIONFUNCTIONTANH_H_ */
