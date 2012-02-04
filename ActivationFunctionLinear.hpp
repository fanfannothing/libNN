/*
 * ActivationFunctionLinear.hpp
 *
 *  Created on: Feb 4, 2012
 *      Author: wchan
 */

#ifndef ACTIVATIONFUNCTIONLINEAR_HPP_
#define ACTIVATIONFUNCTIONLINEAR_HPP_

#include "ActivationFunction.hpp"
#include <iostream>

class ActivationFunctionLinear : public ActivationFunction {
public:
  ActivationFunctionLinear() {
  }
  virtual ~ActivationFunctionLinear() {
  }

  virtual double f(double x) {
    return x;
  }

  virtual double d(double x, double y) {
    return 1;
  }

  virtual double bound(double y) {
    throw;
  }

  virtual void f_cuda(double* x, size_t size) {
    std::cerr << "ActivationFunctionLinear::f_cuda(x) not implemented." << std::endl;
  }

  virtual void d_cuda(double* y, size_t size, double* d) {
    std::cerr << "ActivationFunctionLinear::d_cuda(y, d) not implemented." << std::endl;
  }

  virtual ActivationFunction* clone() {
    // return std::shared_ptr<ActivationFunction>(new ActivationFunctionLinear());
    return new ActivationFunctionLinear();
  }
};

#endif /* ACTIVATIONFUNCTIONLINEAR_HPP_ */
