/*
 * NeuralNetworkLayerConstantCU.hpp
 *
 *  Created on: Jan 31, 2012
 *      Author: wchan
 */

#ifndef NEURALNETWORKLAYERCONSTANTCU_HPP_
#define NEURALNETWORKLAYERCONSTANTCU_HPP_

#include "NeuralNetworkCU.hpp"
#include "NeuralNetworkLayerConstant.hpp"
#include <memory>

class NeuralNetworkLayerConstantCU : public NeuralNetworkCU {
public:
  NeuralNetworkLayerConstantCU(size_t size) {
    m_outputs_size = size;

    cublasAlloc(m_outputs_size, sizeof(double), (void**) &m_outputs);
  }

  NeuralNetworkLayerConstantCU(boost::numeric::ublas::vector<double> constant) {
    m_outputs_size = constant.size();

    cublasAlloc(m_outputs_size, sizeof(double), (void**) &m_outputs);

    cublasSetVector(m_outputs_size, sizeof(double), &constant[0], 1, m_outputs, 1);
  }

  NeuralNetworkLayerConstantCU(std::shared_ptr<NeuralNetworkLayerConstant> network) {
    m_outputs_size = network->get_outputs_size();

    cublasAlloc(m_outputs_size, sizeof(double), (void**) &m_outputs);
  }

  virtual ~NeuralNetworkLayerConstantCU() {
    cublasFree(m_outputs);
  }

  virtual void set_value(boost::numeric::ublas::vector<double> value) {
    cublasSetVector(m_outputs_size, sizeof(double), &value[0], 1, m_outputs, 1);
  }

  virtual void compute() {
  }
};

#endif /* NEURALNETWORKLAYERCONSTANTCU_HPP_ */
