/*
 * ActivationFunctionTanh.h
 *
 *  Created on: Jan 21, 2012
 *      Author: wchan
 */

#ifndef ACTIVATIONFUNCTIONTANH_H_
#define ACTIVATIONFUNCTIONTANH_H_

#include "ActivationFunction.hpp"
#include <memory>
#include <cmath>

class ActivationFunctionTanh : public ActivationFunction {
public:
  ActivationFunctionTanh() {
  }

  virtual ~ActivationFunctionTanh() {
  }

  virtual double f(double x) {
    // double y = 1.7159 * std::tanh(0.666666667 * x);
    double y = std::tanh(x);
    return y;
  }

  virtual double d(double x, double y) {
    // return 0.666666667 / 1.7159 * (1.7159 - y) * (1.7159 + y);
    return 1.0 - y * y;
  }

  virtual double bound(double y) {
    if (y > 0) return 1;
    return -1;
  }

  /* these two puppies need to be implemented in ActivationFunctionTanh.cu due nvcc not supporting C++0x */
  virtual void f_cuda(double* x, size_t size);
  virtual void d_cuda(double* y, size_t size, double* d);

  virtual ActivationFunction* clone() {
    // return std::shared_ptr<ActivationFunction>(new ActivationFunctionTanh());
    return new ActivationFunctionTanh();
  }
};

#endif /* ACTIVATIONFUNCTIONTANH_H_ */
