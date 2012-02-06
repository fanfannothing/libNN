/*
 * NeuralNetworkLayer.hpp
 *
 *  Created on: Jan 21, 2012
 *      Author: wchan
 */

#ifndef NEURALNETWORKLAYER_HPP_
#define NEURALNETWORKLAYER_HPP_

#include "ActivationFunction.hpp"
#include "NeuralNetwork.hpp"
#include <memory>
#include <random>
#include <cmath>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <algorithm>
#include <functional>

class NeuralNetworkLayer : public NeuralNetwork {
public:
  NeuralNetworkLayer(std::size_t count, std::shared_ptr<NeuralNetwork> in, std::shared_ptr<ActivationFunction> activaction) :
      NeuralNetwork(in, activaction) {
    m_weights.resize(count, in->get_outputs_size(), false);
    m_dedw.resize(count, in->get_outputs_size(), false);
    m_dedw_last.resize(count, in->get_outputs_size(), false);
    m_weights_update_value.resize(count, in->get_outputs_size(), false);
    this->m_outputs.resize(count, false);
    this->m_dydx.resize(count, false);

    m_dedw.clear();
    m_dedw_last.clear();

    // Yecun98
    // variance should be mean 0 and standard deviation of sqrt(in_size)
    double std = 1 / std::sqrt(in->get_outputs_size());
    double r = std * 3.46410161514 / 2;

    std::generate(m_weights.data().begin(), m_weights.data().end(), std::bind(std::uniform_real_distribution<double>(-r, r), mt));
  }

  ~NeuralNetworkLayer() {
  }

  virtual void compute() {
    // perform weight calculation
    boost::numeric::ublas::vector<double> activations = prod(m_weights, this->m_prev->get_outputs());

    // element wise activation
    std::transform(activations.begin(), activations.end(), this->m_outputs.begin(), std::bind(&ActivationFunction::f, this->m_activation.get(), std::placeholders::_1));

    // calculate derivative while we are at it...
    std::transform(activations.begin(), activations.end(), this->m_outputs.begin(), this->m_dydx.begin(), std::bind(&ActivationFunction::d, this->m_activation.get(), std::placeholders::_1, std::placeholders::_2));
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
  virtual NeuralNetworkLayer* clone() {
    NeuralNetworkLayer* clone = new NeuralNetworkLayer();

    //clone->m_mse = m_mse;
    //clone->m_outputs = m_outputs;
    //clone->m_dydx = m_dydx;
    //clone->m_dedx = m_dedx;
    clone->m_weights = m_weights;
    //clone->m_weights_update_value = m_weights_update_value;
    //clone->m_dedw = m_dedw;

    clone->m_activation.reset(m_activation->clone());

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
};

#endif /* NEURALNETWORKLAYER_HPP_ */
