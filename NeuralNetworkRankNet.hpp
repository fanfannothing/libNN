/*
 * NeuralNetworkRankNet.hpp
 *
 *  Created on: Feb 4, 2012
 *      Author: wchan
 */

#ifndef NEURALNETWORKRANKNET_HPP_
#define NEURALNETWORKRANKNET_HPP_

#include "ActivationFunctionLinear.hpp"
#include "NeuralNetworkMultilayerPerceptron.hpp"

class NeuralNetworkRankNet : public NeuralNetworkMultilayerPerceptron {
public:
  NeuralNetworkRankNet(std::size_t hidden) :
      NeuralNetworkMultilayerPerceptron( { 136, hidden }) {
    std::shared_ptr<NeuralNetworkLayer> linear(new NeuralNetworkLayer(1, m_layers[m_layers.size() - 1], std::shared_ptr<ActivationFunction>(new ActivationFunctionLinear())));
    add_layer(linear);
  }
};

#endif /* NEURALNETWORKRANKNET_HPP_ */
