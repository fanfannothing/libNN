/*
 * LossFunctionMeanSquaredError.hpp
 *
 *  Created on: Feb 3, 2012
 *      Author: wchan
 */

#ifndef LOSSFUNCTIONMEANSQUAREDERROR_HPP_
#define LOSSFUNCTIONMEANSQUAREDERROR_HPP_

#include <boost/numeric/ublas/vector.hpp>
#include <cublas.h>

class LossFunctionMeanSquaredError {
public:
  static double e(boost::numeric::ublas::vector<double> target, boost::numeric::ublas::vector<double> output, boost::numeric::ublas::vector<double>& dedy) {
    dedy = target - output;

    return 0.5 * norm_2(dedy);
  }

  static double e_cuda(double* target, double* output, size_t count, double* dedy) {
    /* this is to sort of keep a uniform API w. the matrix */
    assert(target == dedy);

    cublasDaxpy(count, -1, output, 1, dedy, 1);

    return 0.5 * cublasDnrm2(count, dedy, 1);
  }
};

#endif /* LOSSFUNCTIONMEANSQUAREDERROR_HPP_ */
