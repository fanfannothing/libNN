/*
 * NeuralNetworkBAM.hpp
 *
 *  Created on: Feb 13, 2012
 *      Author: wchan
 */

#ifndef NEURALNETWORKBAM_HPP_
#define NEURALNETWORKBAM_HPP_

#include "NeuralNetwork.hpp"
#include "ActivationFunctionSgn.hpp"

class NeuralNetworkBAM : public NeuralNetwork {
public:
  NeuralNetworkBAM(std::size_t count_x, std::size_t count_y) {
    m_weights.resize(count_y, count_x, false);
    m_outputs.resize(count_y, false);
    m_activation.reset(new ActivationFunctionSgn());
  }

  virtual ~NeuralNetworkBAM() {
  }

  virtual boost::numeric::ublas::vector<double> f(boost::numeric::ublas::vector<double> in) {
    m_x = in;
    compute();
    return m_outputs;
  }

  virtual void compute() {
    boost::numeric::ublas::vector<double> y = prod(m_weights, m_x);

    /*
     while (!std::equal(output.begin(), output.end(), m_outputs.begin())) {
     m_outputs = output;
     output = prod(m_weights, m_outputs);
     }
     */
  }

  virtual boost::numeric::ublas::matrix<double>& weights() {
    return m_weights;
  }

protected:
  boost::numeric::ublas::matrix<double> m_weights;
  boost::numeric::ublas::vector<double> m_x;
};

#endif /* NEURALNETWORKBAM_HPP_ */
