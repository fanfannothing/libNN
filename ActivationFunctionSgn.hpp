/*
 * ActivationFunctionSgn.hpp
 *
 *  Created on: Feb 13, 2012
 *      Author: wchan
 */

#ifndef ACTIVATIONFUNCTIONSGN_HPP_
#define ACTIVATIONFUNCTIONSGN_HPP_

class ActivationFunctionSgn : public ActivationFunction {
public:
  ActivationFunctionSgn() {
  }
  virtual ~ActivationFunctionSgn() {
  }
  virtual double f(double x) {
    return (x > 0.0) - (x < 0.0);
  }
  virtual double d(double x, double y) {
    return 0;
  }

  virtual ActivationFunction* clone() {
    // return std::shared_ptr<ActivationFunction>(new ActivationFunctionSigmoid());
    return new ActivationFunctionSgn();
  }
};

#endif /* ACTIVATIONFUNCTIONSGN_HPP_ */
