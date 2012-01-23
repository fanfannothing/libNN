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

class ActivationFunctionTanh : public ActivationFunction {
public:
  ActivationFunctionTanh() {
  }
  virtual ~ActivationFunctionTanh() {
  }
  static double f(double x) {
    double y = std::tanh(x);
    y = std::max(y, -0.999);
    y = std::min(y, 0.999);
    return y;
  }
  static double d(double x, double y) {
    return 1 - y * y;
  }
  static double bound(double y) {
    if (y > 0)
      return 1;
    return -1;
  }
};

#endif /* ACTIVATIONFUNCTIONTANH_H_ */
