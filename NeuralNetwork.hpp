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

  double m_mse;
};

#endif /* NEURALNETWORK_HPP_ */
