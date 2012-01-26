/*
 * Backpropagation.hpp
 *
 *  Created on: Jan 21, 2012
 *      Author: wchan
 */

#ifndef BACKPROPAGATION_HPP_
#define BACKPROPAGATION_HPP_

#include "NeuralNetworkMultiLayer.hpp"
#include <algorithm>

template<class ActivationFunction>
class Backpropagation {
public:

  static void train(std::shared_ptr<NeuralNetworkMultiLayer<ActivationFunction> > neural_network, boost::numeric::ublas::vector<double> input, boost::numeric::ublas::vector<double> target, double eta = 0.7) {
    boost::numeric::ublas::vector<double> output = neural_network->f(input);

    std::vector<std::shared_ptr<NeuralNetworkLayer<ActivationFunction> > > layers = neural_network->get_layers();

    // compute the error..
    boost::numeric::ublas::vector<double> dedy = target - output;

    neural_network->mse() += norm_2(dedy);

    // technically dedy is supposed to be a diagonal matrix; but it's a vector in our representation so we do an element wise multiplication
    std::transform(dedy.begin(), dedy.end(), layers[layers.size() - 1]->dydx().begin(), dedy.begin(), std::multiplies<double>());

    // set the error for the last layer
    layers[layers.size() - 1]->dedy() = dedy;

    // backpropagate our dedy...
    for (std::size_t i = layers.size() - 2; i <= layers.size(); i--) {
      std::shared_ptr<NeuralNetwork> current = layers[i];
      std::shared_ptr<NeuralNetwork> next = layers[i + 1];

      boost::numeric::ublas::vector<double> current_errors = prod(trans(next->weights()), next->dedy());

      std::transform(current_errors.begin(), current_errors.end(), current->dydx().begin(), current_errors.begin(), std::multiplies<double>());

      current->dedy() = current_errors;
    }

    // now update our weights...
    for (std::size_t i = 0; i < layers.size(); i++) {
      std::shared_ptr<NeuralNetwork> current = layers[i];
      std::shared_ptr<NeuralNetwork> prev = current->get_inputs();

      current->weights() += eta * outer_prod(current->dedy(), prev->get_outputs());
    }
  }

  static void train(std::shared_ptr<NeuralNetworkMultiLayer<ActivationFunction> > neural_network, std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > labels, double eta = 0.7) {
    neural_network->mse() = 0;
    for (std::size_t i = 0; i < labels.size(); i++) {
      train(neural_network, labels[i].first, labels[i].second, eta);
    }
    neural_network->mse() /= labels.size();
  }
};

#endif /* BACKPROPAGATION_HPP_ */
