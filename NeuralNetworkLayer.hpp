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
    double r = std::sqrt(3 * in->get_outputs_size());

    std::generate(m_weights.data().begin(), m_weights.data().end(), std::bind(std::uniform_real_distribution<double>(-r, r), mt));
  }

  virtual void compute() {
    // perform weight calculation
    boost::numeric::ublas::vector<double> activations = prod(m_weights, m_prev->get_outputs());

    // element wise activation
    std::transform(activations.begin(), activations.end(), m_outputs.begin(), std::ptr_fun(&ActivationFunction::f));

    // calculate derivative while we are at it...
    std::transform(activations.begin(), activations.end(), m_outputs.begin(), m_dydx.begin(), std::ptr_fun(&ActivationFunction::d));
  }

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
    for (std::size_t i = 0; i < m_dedy.size(); i++) {
      std::cerr << m_dedy[i] << " ";
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

protected:
  boost::numeric::ublas::matrix<double> m_weights;
  boost::numeric::ublas::matrix<double> m_weights_update_value;
  boost::numeric::ublas::matrix<double> m_dedw;
  boost::numeric::ublas::matrix<double> m_dedw_last;
};

#endif /* NEURALNETWORKLAYER_HPP_ */
