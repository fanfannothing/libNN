/*
 * ActivationFunction.hpp
 *
 *  Created on: Jan 21, 2012
 *      Author: wchan
 */

#ifndef ACTIVATIONFUNCTION_HPP_
#define ACTIVATIONFUNCTION_HPP_

class ActivationFunction {
public:
  ActivationFunction() {
  }
  virtual ~ActivationFunction() {
  }
  // C++ doesn't allow virtual static functions
  /*
   virtual double f(double x) = 0;
   virtual double d(double y) = 0;
   */
};

#endif /* ACTIVATIONFUNCTION_HPP_ */
