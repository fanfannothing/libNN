/*
 * LossFunctionCrossEntropyError.hpp
 *
 *  Created on: Feb 18, 2012
 *      Author: wchan
 */

#ifndef LOSSFUNCTIONCROSSENTROPYERROR_HPP_
#define LOSSFUNCTIONCROSSENTROPYERROR_HPP_

class LossFunctionCrossEntropyError {
public:
  static double e(boost::numeric::ublas::vector<double> target, boost::numeric::ublas::vector<double> output, boost::numeric::ublas::vector<double>& dedx) {
    dedy = target - output;

    return 0.5 * norm_2(dedy);
  }
};

#endif /* LOSSFUNCTIONCROSSENTROPYERROR_HPP_ */
