/*
 * ActivationFunctionSigmoid.hpp
 *
 *  Created on: Jan 21, 2012
 *      Author: wchan
 */

#ifndef ACTIVATIONFUNCTIONSIGMOID_HPP_
#define ACTIVATIONFUNCTIONSIGMOID_HPP_

#include "ActivationFunction.hpp"
#include <cmath>
#include <algorithm>

class ActivationFunctionSigmoid : public ActivationFunction {
public:
  ActivationFunctionSigmoid() {
  }
  virtual ~ActivationFunctionSigmoid() {
  }
  static double f(double x) {
    double y = 1 / (1 + exp(-2 * x));
    return y;
  }
  static double d(double x, double y) {
    return 2 * y * (1 - y);
  }
  static double bound(double y) {
    if (y > 0.5) return 1;
    return 0;
  }

  static void f_cuda(double* x, size_t size) {
    std::cerr << "ActivationFunctionSigmoid::f_cuda(x) not implemented." << std::endl;
  }
  static void d_cuda(double* y, size_t size, double* d) {
    std::cerr << "ActivationFunctionSigmoid::d_cuda(y, d) not implemented." << std::endl;
  }
};

#endif /* ACTIVATIONFUNCTIONSIGMOID_HPP_ */
