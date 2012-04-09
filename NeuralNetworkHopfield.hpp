/*
 * NeuralNetworkHopfield.hpp
 *
 *  Created on: Feb 13, 2012
 *      Author: wchan
 */

#ifndef NEURALNETWORKHOPFIELD_HPP_
#define NEURALNETWORKHOPFIELD_HPP_

#include "NeuralNetwork.hpp"
#include "ActivationFunctionSgn.hpp"

class NeuralNetworkHopfield : public NeuralNetwork {
public:
  NeuralNetworkHopfield(std::size_t count) {
    m_weights.resize(count, count, false);
    m_outputs.resize(count, false);
    m_activation.reset(new ActivationFunctionSgn());
  }

  virtual void compute() {
    m_outputs = prod(m_weights, m_outputs);
  }

  virtual boost::numeric::ublas::matrix<double>& weights() {
    return m_weights;
  }

protected:
  boost::numeric::ublas::matrix<double> m_weights;
};

#endif /* NEURALNETWORKHOPFIELD_HPP_ */
