/*
 * NeuralNetworkBase.hpp
 *
 *  Created on: Feb 3, 2012
 *      Author: wchan
 */

#ifndef NEURALNETWORKBASE_HPP_
#define NEURALNETWORKBASE_HPP_

#include <memory>
#include <limits>
#include <iostream>
#include <boost/numeric/ublas/vector.hpp>

template<class Vector, class Matrix, class NeuralNetworkType>
class NeuralNetworkBase {
public:
  NeuralNetworkBase() {
    m_error = std::numeric_limits<double>::max();
  }
  NeuralNetworkBase(std::shared_ptr<NeuralNetworkType> in) {
    m_error = std::numeric_limits<double>::max();
    m_prev = in;
  }

  virtual ~NeuralNetworkBase() {
  }

  virtual std::shared_ptr<NeuralNetworkType> get_inputs() {
    return m_prev;
  }

  virtual void set_inputs(std::shared_ptr<NeuralNetworkType> in) {
    m_prev = in;
  }

  virtual std::size_t get_outputs_size() = 0;

  virtual Vector get_outputs() {
    return m_outputs;
  }

  virtual Vector get_outputs(boost::numeric::ublas::vector<double>& outputs) = 0;

  virtual Vector f(boost::numeric::ublas::vector<double> in) {
    std::cerr << "NeuralNetworkBase::f() not supported for this type of neural network." << std::endl;
    throw;
  }

  virtual Vector f(boost::numeric::ublas::vector<double> in, boost::numeric::ublas::vector<double>& outputs) {
    std::cerr << "NeuralNetworkBase::f() not supported for this type of neural network." << std::endl;
    throw;
  }

  virtual Vector& dedx() {
    return m_dedx;
  }

  virtual Vector& dydx() {
    return m_dydx;
  }

  virtual Matrix& weights() {
    std::cerr << "NeuralNetworkBase::weights() not supported for this type of neural network.";
    throw;
  }

  /* used by rprop */
  virtual Matrix& weights_update_value() {
    std::cerr << "NeuralNetworkBase::weights_update_last() not supported for this type of neural network.";
    throw;
  }
  virtual Matrix& dedw_last() {
    std::cerr << "NeuralNetworkBase::dedw_last() not supported for this type of neural network.";
    throw;
  }
  virtual Matrix& dedw() {
    std::cerr << "NeuralNetworkBase::dedw() not supported for this type of neural network.";
    throw;
  }

  virtual void print() {
    std::cerr << "NeuralNetworkBase::print()" << std::endl;
  }

  /* compute the neural network's output based on the input's values */
  virtual void compute() = 0;

  virtual double& error() {
    return m_error;
  }

  virtual NeuralNetworkType* clone() {
    std::cerr << "NeuralNetworkBase::clone() not supported for this type of neural network." << std::endl;
    throw;
  }

protected:
  std::shared_ptr<NeuralNetworkType> m_prev;
  Vector m_outputs;

  /* bottom two mainly used by the backpropagation algorithm... */
  Vector m_dydx;
  Vector m_dedx;

  double m_error;
};

#endif /* NEURALNETWORKBASE_HPP_ */
