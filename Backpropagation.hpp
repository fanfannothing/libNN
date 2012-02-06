/*
 * Backpropagation.hpp
 *
 *  Created on: Jan 21, 2012
 *      Author: wchan
 */

#ifndef BACKPROPAGATION_HPP_
#define BACKPROPAGATION_HPP_

#include "NeuralNetworkMultilayerPerceptron.hpp"
#include "LossFunctionMeanSquaredError.hpp"
#include <algorithm>

template<class LossFunction = LossFunctionMeanSquaredError>
class Backpropagation {
public:

  /**
   * train a single entry... if the input vector passed in is empty (i.e. size 0) the output is taken as is; else it will do a forward pass to recompute the input passed on the input vector passed in
   */
  static void train_single(std::shared_ptr<NeuralNetworkMultilayerPerceptron> neural_network, boost::numeric::ublas::vector<double> input, boost::numeric::ublas::vector<double> target, double eta = 0.001) {
    boost::numeric::ublas::vector<double> output;
    if (input.size() > 0)
      output = neural_network->f(input);
    else
      output = neural_network->get_outputs();

    std::vector<std::shared_ptr<NeuralNetworkLayer> > layers = neural_network->get_layers();

    // compute the error..
    boost::numeric::ublas::vector<double> dedx;
    neural_network->error() += LossFunction::e(target, output, dedx);

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

  static void train_single(std::shared_ptr<NeuralNetworkMultilayerPerceptron> neural_network, std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > labels, double eta = 0.001) {
    neural_network->error() = 0;
    for (std::size_t i = 0; i < labels.size(); i++) {
      train_single(neural_network, labels[i].first, labels[i].second, eta);
    }
    neural_network->error() /= labels.size();
  }

  static void train(std::shared_ptr<NeuralNetworkMultilayerPerceptron> neural_network, std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > labels , std::size_t max_rounds = 100
      , double max_error = 0.001, double eta = 0.001) {
    for (std::size_t i = 0; i != max_rounds && neural_network->error() > max_error; i++) {
      std::cout << "Backprop round " << i;
      train_single(neural_network, labels, eta);
      std::cout << " error: " << neural_network->error() << std::endl;
    }
  }
};

#endif /* BACKPROPAGATION_HPP_ */
