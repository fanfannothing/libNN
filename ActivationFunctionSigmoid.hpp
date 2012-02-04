/*
 * ActivationFunctionSigmoid.hpp
 *
 *  Created on: Jan 21, 2012
 *      Author: wchan
 */

#ifndef ACTIVATIONFUNCTIONSIGMOID_HPP_
#define ACTIVATIONFUNCTIONSIGMOID_HPP_

#include "ActivationFunction.hpp"
#include <memory>
#include <cmath>
#include <algorithm>

class ActivationFunctionSigmoid : public ActivationFunction {
public:
  ActivationFunctionSigmoid() {
  }
  virtual ~ActivationFunctionSigmoid() {
  }
  virtual double f(double x) {
    double y = 1 / (1 + exp(-2 * x));
    return y;
  }
  virtual double d(double x, double y) {
    return 2 * y * (1 - y);
  }
  virtual double bound(double y) {
    if (y > 0.5) return 1;
    return 0;
  }
  virtual void f_cuda(double* x, size_t size) {
    std::cerr << "ActivationFunctionSigmoid::f_cuda(x) not implemented." << std::endl;
  }
  virtual void d_cuda(double* y, size_t size, double* d) {
    std::cerr << "ActivationFunctionSigmoid::d_cuda(y, d) not implemented." << std::endl;
  }
  virtual ActivationFunction* clone() {
    // return std::shared_ptr<ActivationFunction>(new ActivationFunctionSigmoid());
    return new ActivationFunctionSigmoid();
  }
};

#endif /* ACTIVATIONFUNCTIONSIGMOID_HPP_ */
