/*
 * ResilientBackpropagation.hpp
 *
 *  Created on: Jan 22, 2012
 *      Author: wchan
 */

#ifndef RESILIENTBACKPROPAGATION_HPP_
#define RESILIENTBACKPROPAGATION_HPP_

#include "NeuralNetworkMultiLayer.hpp"
#include <algorithm>

/**
 * This is a batch learning algorithm that ignores the sign of the derivative. See the Rprop paper.
 */
template<class ActivationFunction>
class ResilientBackpropagation {
public:
  static double signum(double x) {
    return (x > 0) - (x < 0);
  }

  static void train_batch(std::shared_ptr<NeuralNetworkMultiLayer<ActivationFunction> > neural_network, std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > labels , double eta_minus = 0.5
      , double eta_plus = 1.2, double update_value_min = 1e-6, double update_value_max = 50) {
    neural_network->mse() = 0;

    std::vector<std::shared_ptr<NeuralNetworkLayer<ActivationFunction> > > layers = neural_network->get_layers();
    for (std::size_t i = 0; i < labels.size(); i++) {
      boost::numeric::ublas::vector<double> output = neural_network->f(labels[i].first);

      // compute the error..
      boost::numeric::ublas::vector<double> errors = labels[i].second - output;

      neural_network->mse() += norm_2(errors);

      std::transform(errors.begin(), errors.end(), layers[layers.size() - 1]->derivatives().begin(), errors.begin(), std::multiplies<double>());

      // set the error for the last layer
      layers[layers.size() - 1]->errors() = errors;

      // backpropagate our errors...
      for (std::size_t i = layers.size() - 2; i <= layers.size(); i--) {
        std::shared_ptr<NeuralNetwork> current = layers[i];
        std::shared_ptr<NeuralNetwork> next = layers[i + 1];

        boost::numeric::ublas::vector<double> current_errors = prod(trans(next->weights()), next->errors());

        std::transform(current_errors.begin(), current_errors.end(), current->derivatives().begin(), current_errors.begin(), std::multiplies<double>());

        current->errors() = current_errors;
      }

      for (std::size_t i = 0; i < layers.size(); i++) {
        std::shared_ptr<NeuralNetwork> current = layers[i];
        std::shared_ptr<NeuralNetwork> prev = current->get_inputs();

        current->dedw() += outer_prod(current->errors(), prev->get_outputs());
      }
    }
    neural_network->mse() /= labels.size();

    // now we go ahead and update the weights
    for (std::size_t i = 0; i < layers.size(); i++) {
      std::shared_ptr<NeuralNetwork> current = layers[i];

      for (std::size_t j = 0; j < current->dedw().size1(); j++) {
        for (std::size_t k = 0; k < current->dedw().size2(); k++) {
          double sign = current->dedw()(j, k) * current->dedw_last()(j, k);

          if (sign > 0) {
            current->weights_update_value()(j, k) = std::min(current->weights_update_value()(j, k) * eta_plus, update_value_max);
            current->weights()(j, k) += signum(current->dedw()(j, k)) * current->weights_update_value()(j, k);
            current->dedw_last()(j, k) = current->dedw()(j, k);
          } else if (sign < 0) {
            current->weights_update_value()(j, k) = std::max(current->weights_update_value()(j, k) * eta_minus, update_value_min);
            current->dedw_last()(j, k) = 0;
          } else {
            current->weights()(j, k) += signum(current->dedw()(j, k)) * current->weights_update_value()(j, k);
            current->dedw_last()(j, k) = current->dedw()(j, k);
          }
        }
      }

      current->dedw().clear();
    }
  }

  static void train(std::shared_ptr<NeuralNetworkMultiLayer<ActivationFunction> > neural_network, std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > labels, std::size_t max_rounds = 1000
      , double max_error = 0.01, double initial_weights_update_value = 0.1, double eta_minus = 0.5 , double eta_plus = 1.2, double update_value_min = 1e-6, double update_value_max = 50) {
    std::vector<std::shared_ptr<NeuralNetworkLayer<ActivationFunction> > > layers = neural_network->get_layers();
    for (std::size_t i = 0; i < layers.size(); i++) {
      std::shared_ptr<NeuralNetwork> current = layers[i];
      std::fill(current->weights_update_value().data().begin(), current->weights_update_value().data().end(), initial_weights_update_value);
    }

    for (std::size_t i = 0; i != max_rounds; i++) {
      std::cout << "Rprop round " << i << " mse: " << neural_network->mse() << std::endl;
      train_batch(neural_network, labels, eta_minus, eta_plus, update_value_min, update_value_max);
      if (neural_network->mse() < max_error) {
        i = max_rounds;
        std::cerr << "Rprop done after " << i << std::endl;
      }
    }
  }
};

#endif /* RESILIENTBACKPROPAGATION_HPP_ */
