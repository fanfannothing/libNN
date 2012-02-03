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

class NeuralNetworkCU : public NeuralNetworkBase<double*, double*, NeuralNetworkCU> {
public:
  using NeuralNetworkBase::get_outputs;
  using NeuralNetworkBase::f;

  NeuralNetworkCU() {
  }

  NeuralNetworkCU(std::shared_ptr<NeuralNetworkCU> in) :
      NeuralNetworkBase(in) {
  }

  virtual ~NeuralNetworkCU() {
  }

  virtual size_t get_outputs_size() {
    return m_outputs_size;
  }

  virtual double* get_outputs(boost::numeric::ublas::vector<double>& outputs) {
    if (outputs.size() < m_outputs_size) outputs.resize(m_outputs_size, false);

    cublasGetVector(m_outputs_size, sizeof(double), m_outputs, 1, &outputs[0], 1);

    return get_outputs();
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
    f(in);
    return get_outputs(outputs);
  }

  virtual size_t get_weights_rows() {
    throw;
  }

  virtual size_t get_weights_cols() {
    throw;
  }

  virtual size_t get_weights_lda() {
    throw;
  }

protected:
  size_t m_outputs_size;
};

#endif /* NEURALNETWORKCU_HPP_ */
