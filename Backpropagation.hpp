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

  static void train_single(std::shared_ptr<NeuralNetworkMultiLayer<ActivationFunction> > neural_network, boost::numeric::ublas::vector<double> input, boost::numeric::ublas::vector<double> target, double eta = 0.001) {
    boost::numeric::ublas::vector<double> output = neural_network->f(input);

    std::vector<std::shared_ptr<NeuralNetworkLayer<ActivationFunction> > > layers = neural_network->get_layers();

    // compute the error..
    boost::numeric::ublas::vector<double> dedx = target - output;

    neural_network->mse() += norm_2(dedx);

    // technically dedx is supposed to be a diagonal matrix; but it's a vector in our representation so we do an element wise multiplication
    std::transform(dedx.begin(), dedx.end(), layers[layers.size() - 1]->dydx().begin(), dedx.begin(), std::multiplies<double>());

    // set the error for the last layer
    layers[layers.size() - 1]->dedx() = dedx;

    // backpropagate our dedy...
    for (std::size_t i = layers.size() - 2; i <= layers.size(); i--) {
      std::shared_ptr<NeuralNetwork> current = layers[i];
      std::shared_ptr<NeuralNetwork> next = layers[i + 1];

      boost::numeric::ublas::vector<double> current_dedx = prod(next->dedx(), next->weights());
      std::transform(current_dedx.begin(), current_dedx.end(), current->dydx().begin(), current_dedx.begin(), std::multiplies<double>());

      current->dedx() = current_dedx;
    }

    // now update our weights...
    for (std::size_t i = 0; i < layers.size(); i++) {
      std::shared_ptr<NeuralNetwork> current = layers[i];
      std::shared_ptr<NeuralNetwork> prev = current->get_inputs();

      current->weights() += eta * outer_prod(current->dedx(), prev->get_outputs());
    }
  }

  static void train_single(std::shared_ptr<NeuralNetworkMultiLayer<ActivationFunction> > neural_network, std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > labels, double eta = 0.001) {
    neural_network->mse() = 0;
    for (std::size_t i = 0; i < labels.size(); i++) {
      train_single(neural_network, labels[i].first, labels[i].second, eta);
    }
    neural_network->mse() /= 2 * labels.size();
  }

  static void train(std::shared_ptr<NeuralNetworkMultiLayer<ActivationFunction> > neural_network, std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > labels, std::size_t max_rounds = 100
      , double max_error = 0.001, double eta = 0.001) {
    for (std::size_t i = 0; i != max_rounds && neural_network->mse() > max_error; i++) {
      std::cout << "Backprop round " << i;
      train_single(neural_network, labels, eta);
      std::cout << " mse: " << neural_network->mse() << std::endl;
    }
  }
};

#endif /* BACKPROPAGATION_HPP_ */
