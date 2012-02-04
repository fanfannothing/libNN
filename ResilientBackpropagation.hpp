/*
 * ResilientBackpropagation.hpp
 *
 *  Created on: Jan 22, 2012
 *      Author: wchan
 */

#ifndef RESILIENTBACKPROPAGATION_HPP_
#define RESILIENTBACKPROPAGATION_HPP_

#include "NeuralNetworkMultilayerPerceptron.hpp"
#include <algorithm>
#include <omp.h>

/**
 * This is a batch learning algorithm that ignores the sign of the derivative. See the Rprop paper.
 */
class ResilientBackpropagation {
public:
  static double signum(double x) {
    return (x > 0) - (x < 0);
  }

  static void train_batch(std::shared_ptr<NeuralNetworkMultilayerPerceptron> neural_network, std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > labels , double eta_minus = 0.5
      , double eta_plus = 1.2, double update_value_min = 1e-9, double update_value_max = 10) {
    neural_network->error() = 0;

    std::vector<std::shared_ptr<NeuralNetworkMultilayerPerceptron> > clones;

    for (int i = 0; i < omp_get_max_threads(); i++) {
      clones.push_back(std::shared_ptr<NeuralNetworkMultilayerPerceptron>(neural_network->clone()));
    }

#pragma omp parallel for firstprivate(labels) schedule(static)
    for (std::size_t i = 0; i < labels.size(); i++) {
      std::vector<std::shared_ptr<NeuralNetworkLayer> > layers = clones[omp_get_thread_num()]->get_layers();

      boost::numeric::ublas::vector<double> output = clones[omp_get_thread_num()]->f(labels[i].first);

      // compute the error..
      boost::numeric::ublas::vector<double> dedx = labels[i].second - output;

      clones[omp_get_thread_num()]->error() += norm_2(dedx);

      // technically dedy is supposed to be a diagonal matrix; but it's a vector in our representation so we do an element wise multiplication
      std::transform(dedx.begin(), dedx.end(), layers[layers.size() - 1]->dydx().begin(), dedx.begin(), std::multiplies<double>());

      // set the error for the last layer
      layers[layers.size() - 1]->dedx() = dedx;

      // backpropagate our errors...
      for (std::size_t i = layers.size() - 2; i <= layers.size(); i--) {
        std::shared_ptr<NeuralNetwork> current = layers[i];
        std::shared_ptr<NeuralNetwork> next = layers[i + 1];

        boost::numeric::ublas::vector<double> current_dedx = prod(next->dedx(), next->weights());
        std::transform(current_dedx.begin(), current_dedx.end(), current->dydx().begin(), current_dedx.begin(), std::multiplies<double>());

        current->dedx() = current_dedx;
      }

      for (std::size_t i = 0; i < layers.size(); i++) {
        std::shared_ptr<NeuralNetwork> current = layers[i];
        std::shared_ptr<NeuralNetwork> prev = current->get_inputs();

        current->dedw() += outer_prod(current->dedx(), prev->get_outputs());
      }
    }

    std::vector<std::shared_ptr<NeuralNetworkLayer > > layers = neural_network->get_layers();
    for (int i = 0; i < omp_get_max_threads(); i++) {
      neural_network->error() += clones[i]->error();

      for (std::size_t j = 0; j < layers.size(); j++) {
        layers[j]->dedw() += clones[i]->get_layers()[j]->dedw();
      }
    }
    neural_network->error() /= labels.size();

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

  static void train(std::shared_ptr<NeuralNetworkMultilayerPerceptron> neural_network, std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > labels , std::size_t max_rounds = 100
      , double max_error = 0.001, double initial_weights_update_value = 0.001, double eta_minus = 0.5 , double eta_plus = 1.2, double update_value_min = 1e-9, double update_value_max = 10) {
    std::vector<std::shared_ptr<NeuralNetworkLayer> > layers = neural_network->get_layers();
    for (std::size_t i = 0; i < layers.size(); i++) {
      std::shared_ptr<NeuralNetwork> current = layers[i];
      std::fill(current->weights_update_value().data().begin(), current->weights_update_value().data().end(), initial_weights_update_value);
    }

    for (std::size_t i = 0; i != max_rounds && neural_network->error() > max_error; i++) {
      std::cout << "Rprop round " << i;
      train_batch(neural_network, labels, eta_minus, eta_plus, update_value_min, update_value_max);
      std::cout << " error: " << neural_network->error() << std::endl;
    }
  }
};

#endif /* RESILIENTBACKPROPAGATION_HPP_ */
