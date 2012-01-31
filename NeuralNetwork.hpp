/*
 * NeuralNetwork.hpp
 *
 *  Created on: Jan 21, 2012
 *      Author: wchan
 */

#ifndef NEURALNETWORK_HPP_
#define NEURALNETWORK_HPP_

#include <memory>
#include <random>
#include <limits>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

#ifdef LIBNNCUDA
#include <cuda_runtime.h>
#include <cublas.h>
#endif

static std::mt19937 mt;

class NeuralNetwork {
public:
  NeuralNetwork() {
    m_mse = std::numeric_limits<double>::max();
  }
  NeuralNetwork(std::shared_ptr<NeuralNetwork> in) {
    m_mse = std::numeric_limits<double>::max();
    m_prev = in;
  }

  virtual ~NeuralNetwork() {
  }

  virtual std::shared_ptr<NeuralNetwork> get_inputs() {
    return m_prev;
  }
  /*
   virtual std::shared_ptr<NeuralNetwork> get_output() {
   return m_next;
   }
   */
  virtual void set_inputs(std::shared_ptr<NeuralNetwork> in) {
    m_prev = in;
  }
  /*
   virtual void set_output(std::shared_ptr<NeuralNetwork> out) {
   m_next = out;
   }
   */

  virtual std::size_t get_outputs_size() {
    return m_outputs.size();
  }

  /* compiler's are smart enough; no need to return constant reference... see Return-value optimization */
  virtual boost::numeric::ublas::vector<double> get_outputs() {
    return m_outputs;
  }
  virtual boost::numeric::ublas::vector<double> f(boost::numeric::ublas::vector<double> in) {
    std::cerr << "NeuralNetwork::f() not supported for this type of neural network." << std::endl;
    return {};
  }

  virtual boost::numeric::ublas::vector<double>& dedx() {
    return m_dedx;
  }
  virtual boost::numeric::ublas::vector<double>& dydx() {
    return m_dydx;
  }
  virtual boost::numeric::ublas::matrix<double>& weights() {
    std::cerr << "NeuralNetwork::weights() not supported for this type of neural network.";
    throw;
  }
  /* used by rprop */
  virtual boost::numeric::ublas::matrix<double>& weights_update_value() {
    std::cerr << "NeuralNetwork::weights_update_last() not supported for this type of neural network.";
    throw;
  }
  virtual boost::numeric::ublas::matrix<double>& dedw_last() {
    std::cerr << "NeuralNetwork::dedw_last() not supported for this type of neural network.";
    throw;
  }
  virtual boost::numeric::ublas::matrix<double>& dedw() {
    std::cerr << "NeuralNetwork::dedw() not supported for this type of neural network.";
    throw;
  }

  virtual void print() {
    std::cerr << "NeuralNetwork::print()" << std::endl;
  }

  /* compute the neural network's output based on the input's values */
  virtual void compute() = 0;

#ifdef LIBNNCUDA
  virtual double* f_cuda(boost::numeric::ublas::vector<double> in) {
    std::cerr << "NeuralNetwork::f_cuda() not supported for this type of neural network." << std::endl;
    return NULL;
  }

  virtual void copy_outputs_cuda_to_host() {
    cublasGetVector(m_outputs_size_cuda, sizeof(double), m_outputs_cuda, 1, &m_outputs[0], 1);
    // cudaMemcpy(&m_outputs[0], m_outputs_cuda, m_outputs_size_cuda * sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&m_dydx[0], m_dydx_cuda, m_outputs_size_cuda, cudaMemcpyDeviceToHost);
    // cudaMemcpy(&m_dedx[0], m_dedx_cuda, m_outputs_size_cuda, cudaMemcpyDeviceToHost);
  }

  virtual void copy_weights_cuda_to_host() {
    assert(weights().size1() == get_weights_rows_cuda());
    assert(weights().size2() == get_weights_cols_cuda());

    double* tmp = new double[weights().size1() * weights().size2()];
    cublasGetMatrix(weights().size1(), weights().size2(), sizeof(double), get_weights_cuda(), get_weights_lda_cuda(), tmp, weights().size1());

    for (size_t i = 0; i < weights().size2(); i++) {
      for (size_t j = 0; j < weights().size1(); j++) {
        weights()(j, i) = tmp[i * weights().size1() + j];
      }
    }

    delete[] tmp;
  }

  virtual void copy_weights_host_to_cuda() {
    assert(weights().size1() == get_weights_rows_cuda());
    assert(weights().size2() == get_weights_cols_cuda());

    double* tmp = new double[weights().size1() * weights().size2()];
    for (size_t i = 0; i < weights().size2(); i++) {
      for (size_t j = 0; j < weights().size1(); j++) {
        tmp[i * weights().size1() + j] = weights()(j, i);
      }
    }

    cublasSetMatrix(weights().size1(), weights().size2(), sizeof(double), tmp, weights().size1(), get_weights_cuda(), get_weights_lda_cuda());

    delete[] tmp;
  }

  /* compute_cuda is different than compute, the results are stored in VRAM rather than system memory */
  virtual void compute_cuda() {
    std::cerr << "NeuralNetwork::compute_cuda() not supported for this type of neural network." << std::endl;
  }

  virtual double* get_outputs_cuda() {
    return m_outputs_cuda;
  }

  virtual double* get_dydx_cuda() {
    return m_dydx_cuda;
  }

  virtual double* get_dedx_cuda() {
    return m_dedx_cuda;
  }

  virtual double* get_weights_cuda() {
    throw;
  }

  virtual size_t get_weights_rows_cuda() {
    throw;
  }

  virtual size_t get_weights_cols_cuda() {
    throw;
  }

  virtual size_t get_weights_lda_cuda() {
    throw;
  }
#endif

  virtual double& mse() {
    return m_mse;
  }

  virtual NeuralNetwork* clone() {
    std::cerr << "NeuralNetwork::clone() not supported for this type of neural network." << std::endl;

    return NULL;
  }

protected:
  std::shared_ptr<NeuralNetwork> m_prev;
// std::shared_ptr<NeuralNetwork> m_next;
  boost::numeric::ublas::vector<double> m_outputs;

  /* bottom two mainly used by the backpropagation algorithm... */
  boost::numeric::ublas::vector<double> m_dydx;
  boost::numeric::ublas::vector<double> m_dedx;

#ifdef LIBNNCUDA
  double* m_outputs_cuda;
  size_t m_outputs_size_cuda;
  size_t m_outputs_pitch_cuda;

  double* m_dydx_cuda;
  double* m_dedx_cuda;
#endif

  double m_mse;
};

#endif /* NEURALNETWORK_HPP_ */
