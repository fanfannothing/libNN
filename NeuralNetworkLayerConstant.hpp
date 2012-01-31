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

#ifdef LIBNNCUDA
    m_outputs_size_cuda = size;

    cublasAlloc(m_outputs_size_cuda, sizeof(double), (void**)&m_outputs_cuda);
    //cudaMallocPitch(&m_outputs_cuda, &m_outputs_pitch_cuda, m_outputs_size_cuda * sizeof(double), 1);
    //cudaMallocPitch(&m_dydx_cuda, &m_outputs_pitch_cuda, m_outputs_size_cuda * sizeof(double), 1);
    //cudaMallocPitch(&m_dedx_cuda, &m_outputs_pitch_cuda, m_outputs_size_cuda * sizeof(double), 1);
#endif
  }

  NeuralNetworkLayerConstant(boost::numeric::ublas::vector<double> constant) {
    m_outputs = constant;

#ifdef LIBNNCUDA
    m_outputs_size_cuda = constant.size();

    cublasAlloc(m_outputs_size_cuda, sizeof(double), (void**)&m_outputs_cuda);
    //cudaMallocPitch(&m_dydx_cuda, &m_outputs_pitch_cuda, m_outputs_size_cuda * sizeof(double), 1);
    //cudaMallocPitch(&m_dedx_cuda, &m_outputs_pitch_cuda, m_outputs_size_cuda * sizeof(double), 1);

    cudaMemcpy(m_outputs_cuda, &constant[0], m_outputs_size_cuda * sizeof(double), cudaMemcpyHostToDevice);
#endif
  }

  virtual ~NeuralNetworkLayerConstant() {
#ifdef LIBNNCUDA
    cublasFree(m_outputs_cuda);
#endif
  }

  virtual void set_value(boost::numeric::ublas::vector<double> value) {
    assert(value.size() == m_outputs.size());

    m_outputs = value;
  }

#ifdef LIBNNCUDA
  virtual void set_value_cuda(boost::numeric::ublas::vector<double> value) {
    cudaMemcpy(m_outputs_cuda, &value[0], m_outputs_size_cuda * sizeof(double), cudaMemcpyHostToDevice);
  }

  virtual void compute_cuda() {}
#endif

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
#ifdef LIBNNCUDA
#warning NeuralNetworkLayerConstant::clone() not implemented for CUDA
#endif
    NeuralNetworkLayerConstant* clone = new NeuralNetworkLayerConstant(m_outputs.size());
    clone->m_outputs = m_outputs;
    return clone;
  }
};

#endif /* NEURALNETWORKLAYERCONSTANT_HPP_ */
