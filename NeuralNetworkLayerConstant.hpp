/*
 * NeuralNetworkLayerConstant.hpp
 *
 *  Created on: Jan 21, 2012
 *      Author: wchan
 */

#ifndef NEURALNETWORKLAYERCONSTANT_HPP_
#define NEURALNETWORKLAYERCONSTANT_HPP_

#include "NeuralNetwork.hpp"
#include <memory>

class NeuralNetworkLayerConstant : public NeuralNetwork {
public:
  NeuralNetworkLayerConstant(std::size_t size) {
    m_outputs.resize(size, false);
  }

  NeuralNetworkLayerConstant(boost::numeric::ublas::vector<double> constant) {
    m_outputs = constant;
  }

  virtual void set_value(boost::numeric::ublas::vector<double> value) {
    assert(value.size() == m_outputs.size());

    m_outputs = value;
  }

  virtual void compute() {
  }

  virtual void print() {
    std::cerr << "m_outputs:\t";
    for (std::size_t i = 0; i < m_outputs.size(); i++) {
      std::cerr << m_outputs[i] << " ";
    }
    std::cerr << std::endl;
  }

  virtual NeuralNetworkLayerConstant* clone() {
    NeuralNetworkLayerConstant* clone = new NeuralNetworkLayerConstant(m_outputs.size());
    clone->m_outputs = m_outputs;
    return clone;
  }
};

#endif /* NEURALNETWORKLAYERCONSTANT_HPP_ */
