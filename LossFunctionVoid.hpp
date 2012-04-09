/*
 * LossFunctionVoid.hpp
 *
 *  Created on: Feb 5, 2012
 *      Author: wchan
 */

#ifndef LOSSFUNCTIONVOID_HPP_
#define LOSSFUNCTIONVOID_HPP_

/**
 * This is a dummy loss function that does nothing
 */
class LossFunctionVoid {
public:
  static double e(boost::numeric::ublas::vector<double> target, boost::numeric::ublas::vector<double> output, boost::numeric::ublas::vector<double>& dedy) {
    return 0;
  }

  static double e_cuda(double* target, double* output, size_t count, double* dedy) {
    return 0;
  }
};

#endif /* LOSSFUNCTIONVOID_HPP_ */
