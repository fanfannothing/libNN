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
};

#endif /* ACTIVATIONFUNCTIONSIGMOID_HPP_ */
