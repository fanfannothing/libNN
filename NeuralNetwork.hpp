/*
 * NeuralNetwork.hpp
 *
 *  Created on: Jan 21, 2012
 *      Author: wchan
 */

#ifndef NEURALNETWORK_HPP_
#define NEURALNETWORK_HPP_

#include "NeuralNetworkBase.hpp"
#include <memory>
#include <random>
#include <limits>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

class NeuralNetwork : public NeuralNetworkBase<boost::numeric::ublas::vector<double>, boost::numeric::ublas::matrix<double>, NeuralNetwork> {
public:
  using NeuralNetworkBase::get_outputs;
  using NeuralNetworkBase::f;

  NeuralNetwork() {
  }

  NeuralNetwork(std::shared_ptr<NeuralNetwork> in, std::shared_ptr<ActivationFunction> activation) :
      NeuralNetworkBase(in, activation) {
  }

  virtual ~NeuralNetwork() {
  }

  virtual std::size_t get_outputs_size() {
    return this->m_outputs.size();
  }

  virtual boost::numeric::ublas::vector<double> get_outputs(boost::numeric::ublas::vector<double>& outputs) {
    outputs = get_outputs();
    return outputs;
  }

  virtual boost::numeric::ublas::vector<double> f(boost::numeric::ublas::vector<double> in, boost::numeric::ublas::vector<double>& outputs) {
    boost::numeric::ublas::vector<double> out = f(in);
    outputs = out;
    return out;
  }

};

#endif /* NEURALNETWORK_HPP_ */
