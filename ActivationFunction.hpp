/*
 * ActivationFunction.hpp
 *
 *  Created on: Jan 21, 2012
 *      Author: wchan
 */

#ifndef ACTIVATIONFUNCTION_HPP_
#define ACTIVATIONFUNCTION_HPP_

#include <memory>
#include <iostream>
#include <stdint.h>

class ActivationFunction {
public:
  ActivationFunction() {
  }
  virtual ~ActivationFunction() {
  }

  virtual double f(double x) = 0;
  virtual double d(double x, double y) = 0;

  virtual double bound(double y) {
    throw;
  }

  virtual void f_cuda(double* x, size_t size) {
    std::cerr << "ActivationFunction::f_cuda not supported." << std::endl;
    throw;
  }

  virtual void d_cuda(double* y, size_t size, double* d) {
    std::cerr << "ActivationFunction::d_cuda not supported." << std::endl;
    throw;
  }

  /* can't use std::shared_ptr here because nvcc doesn't support it...*/
  virtual ActivationFunction* clone()  = 0;
};

#endif /* ACTIVATIONFUNCTION_HPP_ */
