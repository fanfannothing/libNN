/*
 * NeuralNetworkFunctionApproximator.hpp
 *
 *  Created on: Feb 18, 2012
 *      Author: wchan
 */

#ifndef NEURALNETWORKFUNCTIONAPPROXIMATOR_HPP_
#define NEURALNETWORKFUNCTIONAPPROXIMATOR_HPP_

#include "NeuralNetworkMultilayerPerceptron.hpp"
#include "NeuralNetworkLayer.hpp"
#include "ActivationFunctionLinear.hpp"

class NeuralNetworkFunctionApproximator : public NeuralNetworkMultilayerPerceptron {
public:
  NeuralNetworkFunctionApproximator(std::size_t input, std::size_t hidden, std::size_t output) :
      NeuralNetworkMultilayerPerceptron( { input, hidden }) {
    std::shared_ptr<NeuralNetworkLayer> linear(new NeuralNetworkLayer(output, get_layers()[get_layers().size() - 1], std::shared_ptr<ActivationFunction>(new ActivationFunctionLinear())));
    add_layer(linear);
  }

};

#endif /* NEURALNETWORKFUNCTIONAPPROXIMATOR_HPP_ */
