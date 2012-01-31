/*
 * NeuralNetworkLayer.hpp
 *
 *  Created on: Jan 21, 2012
 *      Author: wchan
 */

#ifndef NEURALNETWORKLAYER_HPP_
#define NEURALNETWORKLAYER_HPP_

#include "NeuralNetwork.hpp"
#include <memory>
#include <random>
#include <cmath>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <algorithm>
#include <functional>

#ifdef LIBNNCUDA
#include <cuda_runtime.h>
#include <cublas.h>
#endif

template<class ActivationFunction>
class NeuralNetworkLayer : public NeuralNetwork {
public:
  NeuralNetworkLayer(std::size_t count, std::shared_ptr<NeuralNetwork> in) :
      NeuralNetwork(in) {
    m_weights.resize(count, in->get_outputs_size(), false);
    m_dedw.resize(count, in->get_outputs_size(), false);
    m_dedw_last.resize(count, in->get_outputs_size(), false);
    m_weights_update_value.resize(count, in->get_outputs_size(), false);
    m_outputs.resize(count, false);
    m_dydx.resize(count, false);

    m_dedw.clear();
    m_dedw_last.clear();

    // Yecun98
    // variance should be mean 0 and standard deviation of sqrt(in_size)
    double std = 1 / std::sqrt(in->get_outputs_size());
    double r = std * 3.46410161514 / 2;

    std::generate(m_weights.data().begin(), m_weights.data().end(), std::bind(std::uniform_real_distribution<double>(-r, r), mt));

#ifdef LIBNNCUDA
    m_outputs_size_cuda = count;
    m_weights_m_cuda = count;
    m_weights_n_cuda = in->get_outputs_size();

    // TODO: fix alignment for more performance later
    // careful in allocation... CUDA stores matrices in column major format
    cublasAlloc(m_weights_m_cuda * m_weights_n_cuda, sizeof(double), (void**)&m_weights_cuda);
    cublasAlloc(m_outputs_size_cuda, sizeof(double), (void**)&m_outputs_cuda);
    cublasAlloc(m_outputs_size_cuda, sizeof(double), (void**)&m_dydx_cuda);
    cublasAlloc(m_outputs_size_cuda, sizeof(double), (void**)&m_dedx_cuda);

    // lda is number of rows since the matrices are stored in col-major format
    m_weights_lda_cuda = m_weights_m_cuda;

    // TODO: need to initalize the weights (i.e. on CUDA instead of copy CPU hack right now)
    copy_weights_host_to_cuda();

    boost::numeric::ublas::matrix<double> t(m_weights);
    copy_weights_cuda_to_host();

    assert(std::equal(t.data().begin(), t.data().end(), m_weights.data().begin()));
#endif
  }

  ~NeuralNetworkLayer() {
#ifdef LIBNNCUDA
    cublasFree(m_weights_cuda);
    cublasFree(m_outputs_cuda);
    cublasFree(m_dydx_cuda);
    cublasFree(m_dedx_cuda);
#endif
  }

  virtual void compute() {
    // perform weight calculation
    boost::numeric::ublas::vector<double> activations = prod(m_weights, m_prev->get_outputs());

    // element wise activation
    std::transform(activations.begin(), activations.end(), m_outputs.begin(), std::ptr_fun(&ActivationFunction::f));

    // calculate derivative while we are at it...
    std::transform(activations.begin(), activations.end(), m_outputs.begin(), m_dydx.begin(), std::ptr_fun(&ActivationFunction::d));
  }

#ifdef LIBNNCUDA
  virtual void compute_cuda() {
    cublasDgemv('N', m_weights_m_cuda, m_weights_n_cuda, 1.0, m_weights_cuda, m_weights_lda_cuda, m_prev->get_outputs_cuda(), 1, 0, m_outputs_cuda, 1);

    ActivationFunction::f_cuda(m_outputs_cuda, m_outputs_size_cuda);
    ActivationFunction::d_cuda(m_outputs_cuda, m_outputs_size_cuda, m_dydx_cuda);
  }

  virtual double* get_weights_cuda() {
    return m_weights_cuda;
  }

  virtual size_t get_weights_rows_cuda() {
    return m_weights_m_cuda;
  }

  virtual size_t get_weights_cols_cuda() {
    return m_weights_n_cuda;
  }

  virtual size_t get_weights_lda_cuda() {
    return m_weights_lda_cuda;
  }
#endif

  virtual void print() {
    std::cerr << "m_weights:\t";
    for (std::size_t i = 0; i < m_weights.size1(); i++) {
      for (std::size_t j = 0; j < m_weights.size2(); j++) {
        std::cerr << m_weights(i, j) << " ";
      }
    }
    std::cerr << std::endl;

    std::cerr << "m_outputs:\t";
    for (std::size_t i = 0; i < m_outputs.size(); i++) {
      std::cerr << m_outputs[i] << " ";
    }
    std::cerr << std::endl;

    std::cerr << "m_errors:\t";
    for (std::size_t i = 0; i < m_dedx.size(); i++) {
      std::cerr << m_dedx[i] << " ";
    }
    std::cerr << std::endl;
  }

  virtual boost::numeric::ublas::matrix<double>& weights() {
    return m_weights;
  }
  virtual boost::numeric::ublas::matrix<double>& weights_update_value() {
    return m_weights_update_value;
  }
  virtual boost::numeric::ublas::matrix<double>& dedw() {
    return m_dedw;
  }
  virtual boost::numeric::ublas::matrix<double>& dedw_last() {
    return m_dedw_last;
  }
  virtual NeuralNetworkLayer<ActivationFunction>* clone() {
    NeuralNetworkLayer<ActivationFunction>* clone = new NeuralNetworkLayer<ActivationFunction>();

    //clone->m_mse = m_mse;
    clone->m_outputs = m_outputs;
    clone->m_dydx = m_dydx;
    clone->m_dedx = m_dedx;
    clone->m_weights = m_weights;
    clone->m_weights_update_value = m_weights_update_value;
    clone->m_dedw = m_dedw;
    clone->m_dedw_last = m_dedw_last;

    return clone;
  }

protected:
  NeuralNetworkLayer() {
  }

  boost::numeric::ublas::matrix<double> m_weights;

  /* used by rprop */
  boost::numeric::ublas::matrix<double> m_weights_update_value;
  boost::numeric::ublas::matrix<double> m_dedw;
  boost::numeric::ublas::matrix<double> m_dedw_last;

#ifdef LIBNNCUDA
  double* m_weights_cuda;
  size_t m_weights_m_cuda; /* row */
  size_t m_weights_n_cuda; /* col */
  size_t m_weights_lda_cuda; /* assert(m_weights_lda_cuda == m_weights_pitch_cuda / sizeof(double)) */
  size_t m_weights_pitch_cuda;
#endif
};

#endif /* NEURALNETWORKLAYER_HPP_ */
