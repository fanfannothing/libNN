/*
 * NeuralNetworkLayerCU.hpp
 *
 *  Created on: Jan 31, 2012
 *      Author: wchan
 */

#ifndef NEURALNETWORKLAYERCU_HPP_
#define NEURALNETWORKLAYERCU_HPP_

#include "NeuralNetworkLayer.hpp"
#include "NeuralNetworkCU.hpp"
#include <memory>
#include <random>
#include <cmath>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <algorithm>
#include <functional>
#include <cuda_runtime.h>
#include <cublas.h>

class NeuralNetworkLayerCU : public NeuralNetworkCU {
public:
  NeuralNetworkLayerCU(size_t count, std::shared_ptr<NeuralNetworkCU> in, std::shared_ptr<ActivationFunction> activation) :
      NeuralNetworkCU(in, activation) {
    m_outputs_size = count;
    m_weights_m = count;
    m_weights_n = in->get_outputs_size();

    // lda is number of rows since the matrices are stored in col-major format
    m_weights_lda = m_weights_m;

    // TODO: fix alignment for more performance later
    // careful in allocation... CUDA stores matrices in column major format
    cublasAlloc(m_weights_m * m_weights_n, sizeof(double), (void**) &m_weights);
    cublasAlloc(m_outputs_size, sizeof(double), (void**) &m_outputs);
    cublasAlloc(m_outputs_size, sizeof(double), (void**) &m_dydx);
    cublasAlloc(m_outputs_size, sizeof(double), (void**) &m_dedx);

    // TODO: need to initalize the weights (i.e. on CUDA instead of copy CPU hack right now)
    // Yecun98
    // variance should be mean 0 and standard deviation of sqrt(in_size)
    double std = 1 / std::sqrt(in->get_outputs_size());
    double r = std * 3.46410161514 / 2;

    boost::numeric::ublas::matrix<double> w(m_weights_m, m_weights_n);
    std::generate(w.data().begin(), w.data().end(), std::bind(std::uniform_real_distribution<double>(-r, r), mt));

    set_weights(w);
  }

  NeuralNetworkLayerCU(std::shared_ptr<NeuralNetworkLayer> network, std::shared_ptr<NeuralNetworkCU> in) :
      NeuralNetworkCU(in, std::shared_ptr<ActivationFunction>(network->get_activation_function()->clone())) {
    m_outputs_size = network->get_outputs_size();
    m_weights_m = network->get_outputs_size();
    m_weights_n = in->get_outputs_size();

    // lda is number of rows since the matrices are stored in col-major format
    m_weights_lda = m_weights_m;

    // TODO: fix alignment for more performance later
    // careful in allocation... CUDA stores matrices in column major format
    cublasAlloc(m_weights_m * m_weights_n, sizeof(double), (void**) &m_weights);
    cublasAlloc(m_outputs_size, sizeof(double), (void**) &m_outputs);
    cublasAlloc(m_outputs_size, sizeof(double), (void**) &m_dydx);
    cublasAlloc(m_outputs_size, sizeof(double), (void**) &m_dedx);

    set_weights(network->weights());
  }

  ~NeuralNetworkLayerCU() {
    cublasFree(m_weights);
    cublasFree(m_outputs);
    cublasFree(m_dydx);
    cublasFree(m_dedx);
  }

  virtual void compute() {
    cublasDgemv('N', m_weights_m, m_weights_n, 1.0, m_weights, m_weights_lda, m_prev->get_outputs(), 1, 0, m_outputs, 1);

    m_activation->f_cuda(m_outputs, m_outputs_size);
    m_activation->d_cuda(m_outputs, m_outputs_size, m_dydx);
  }

  virtual size_t get_weights_rows() {
    return m_weights_m;
  }

  virtual size_t get_weights_cols() {
    return m_weights_n;
  }

  virtual size_t get_weights_lda() {
    return m_weights_lda;
  }

  virtual double*& weights() {
    return m_weights;
  }

protected:
  NeuralNetworkLayerCU() {
  }

  double* m_weights;

  size_t m_weights_m; /* row */
  size_t m_weights_n; /* col */
  size_t m_weights_lda; /* assert(m_weights_lda == m_weights_pitch / sizeof(double)) */
};

#endif /* NEURALNETWORKLAYERCU_HPP_ */
