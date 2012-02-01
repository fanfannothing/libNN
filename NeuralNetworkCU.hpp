/*
 * NeuralNetworkCU.hpp
 *
 *  Created on: Jan 31, 2012
 *      Author: wchan
 */

#ifndef NEURALNETWORKCU_HPP_
#define NEURALNETWORKCU_HPP_

#include <memory>
#include <random>
#include <limits>
#include <cassert>
#include <cuda_runtime.h>
#include <cublas.h>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

static std::mt19937 mtcu;

class NeuralNetworkCU {
public:
  NeuralNetworkCU() {
    m_mse = std::numeric_limits<double>::max();
  }

  NeuralNetworkCU(std::shared_ptr<NeuralNetworkCU> in) {
    m_mse = std::numeric_limits<double>::max();
    m_prev = in;
  }

  virtual ~NeuralNetworkCU() {
  }

  virtual std::shared_ptr<NeuralNetworkCU> get_inputs() {
    return m_prev;
  }

  virtual void set_inputs(std::shared_ptr<NeuralNetworkCU> in) {
    m_prev = in;
  }

  virtual size_t get_outputs_size() {
    return m_outputs_size;
  }

  virtual double* get_outputs() {
    return m_outputs;
  }

  virtual void get_outputs(boost::numeric::ublas::vector<double>& outputs) {
    if (outputs.size() < m_outputs_size) outputs.resize(m_outputs_size, false);

    cublasGetVector(m_outputs_size, sizeof(double), m_outputs, 1, &outputs[0], 1);
  }

  virtual double* weights() {
    std::cerr << "NeuralNetwork::weights() not supported for this type of neural network." << std::endl;
    throw;
  }

  virtual void get_weights(boost::numeric::ublas::matrix<double>& w) {
    assert(w.size1() == get_weights_rows());
    assert(w.size2() == get_weights_cols());

    double* tmp = new double[w.size1() * w.size2()];

    cublasGetMatrix(w.size1(), w.size2(), sizeof(double), weights(), get_weights_lda(), tmp, w.size1());

    // need to do a transpose
    for (size_t i = 0; i < w.size2(); i++) {
      for (size_t j = 0; j < w.size1(); j++) {
        w(j, i) = tmp[i * w.size1() + j];
      }
    }

    delete[] tmp;
  }

  virtual void set_weights(boost::numeric::ublas::matrix<double> w) {
    assert(w.size1() == get_weights_rows());
    assert(w.size2() == get_weights_cols());

    double* tmp = new double[w.size1() * w.size2()];
    for (size_t i = 0; i < w.size2(); i++) {
      for (size_t j = 0; j < w.size1(); j++) {
        tmp[i * w.size1() + j] = w(j, i);
      }
    }

    cublasSetMatrix(w.size1(), w.size2(), sizeof(double), tmp, w.size1(), weights(), get_weights_lda());

    delete[] tmp;
  }

  virtual double* f(boost::numeric::ublas::vector<double> in) {
    std::cerr << "NeuralNetwork::f() not supported for this type of neural network." << std::endl;
    throw;
  }

  virtual double* f(boost::numeric::ublas::vector<double> in, boost::numeric::ublas::vector<double>& outputs) {
    std::cerr << "NeuralNetwork::f(,) not supported for this type of neural network." << std::endl;
    throw;
  }

  virtual double* dedx() {
    return m_dedx;
  }

  virtual double* dydx() {
    return m_dydx;
  }

  /* compute the neural network's output based on the input's values */
  virtual void compute() = 0;

  virtual size_t get_weights_rows() {
    throw;
  }

  virtual size_t get_weights_cols() {
    throw;
  }

  virtual size_t get_weights_lda() {
    throw;
  }

  virtual double& mse() {
    return m_mse;
  }

protected:
  std::shared_ptr<NeuralNetworkCU> m_prev;

  double* m_outputs;
  size_t m_outputs_size;

  double* m_dydx;
  double* m_dedx;

  double m_mse;
};

#endif /* NEURALNETWORKCU_HPP_ */
