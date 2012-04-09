/*
 * LossFunctionRankNet.hpp
 *
 *  Created on: Feb 5, 2012
 *      Author: wchan
 */

#ifndef LOSSFUNCTIONRANKNET_HPP_
#define LOSSFUNCTIONRANKNET_HPP_

/**
 * This is a hacky loss function... we need to refactor Backpropagation and how the loss function is handled to fix this
 *
 * Basically the "target" is actually the probabilistic loss function in the RankNet paper
 *
 * The real target is 1.0 constant (i.e. we always want to be 100% sure about our ordered pair ranking)
 *
 * We also assume a size of 1 for the vectors
 */
class LossFunctionRankNet {
public:
  static double e(boost::numeric::ublas::vector<double> target, boost::numeric::ublas::vector<double> output, boost::numeric::ublas::vector<double>& dedy) {
    dedy[0] = 1.0 - target[0];

    return 0;
  }

  static double e_cuda(double* target, double* output, size_t count, double* dedy) {
    return 0;
  }
};

#endif /* LOSSFUNCTIONRANKNET_HPP_ */
