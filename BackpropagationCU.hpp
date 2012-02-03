/*
 * BackpropagationCU.hpp
 *
 *  Created on: Jan 29, 2012
 *      Author: wchan
 */

#ifndef BACKPROPAGATIONCU_HPP_
#define BACKPROPAGATIONCU_HPP_

#include "NeuralNetworkMultilayerPerceptronCU.hpp"
#include "LossFunctionMeanSquaredError.hpp"
#include <algorithm>
#include <cublas.h>

/* NeuralNetworkMultiLayerCU.cu :: x = x * y */
void mult(double* x, double* y, size_t count);

template<class ActivationFunction, class LossFunction = LossFunctionMeanSquaredError>
class BackpropagationCU {
public:

  static void train_single(std::shared_ptr<NeuralNetworkMultilayerPerceptronCU<ActivationFunction> > neural_network, boost::numeric::ublas::vector<double> input, boost::numeric::ublas::vector<double> target, double eta = 0.001) {
    double* output = neural_network->f(input);

    std::vector<std::shared_ptr<NeuralNetworkLayerCU<ActivationFunction> > > layers = neural_network->get_layers();

    double* dedx = layers[layers.size() - 1]->dedx();

    // dedx = target
    cublasSetVector(target.size(), sizeof(double), &target[0], 1, dedx, 1);

    // error calculation
    neural_network->mse() += LossFunction::e_cuda(dedx, output, target.size(), dedx);

    // element-wise multiplication with the dydx
    mult(dedx, layers[layers.size() - 1]->dydx(), target.size());

    // backpropgate our dedy
    for (std::size_t i = layers.size() - 2; i <= layers.size(); i--) {
      std::shared_ptr<NeuralNetworkCU> current = layers[i];
      std::shared_ptr<NeuralNetworkCU> next = layers[i + 1];

      double* current_dedx = current->dedx();
      cublasDgemv('T', next->get_weights_rows(), next->get_weights_cols(), 1.0, next->weights(), next->get_weights_lda(), next->dedx(), 1, 0, current_dedx, 1);

      mult(current_dedx, current->dydx(), current->get_outputs_size());
    }

    // now update our weights...
    for (std::size_t i = 0; i < layers.size(); i++) {
      std::shared_ptr<NeuralNetworkCU> current = layers[i];
      std::shared_ptr<NeuralNetworkCU> prev = current->get_inputs();

      cublasDger(current->get_weights_rows(), current->get_weights_cols(), eta, current->dedx(), 1, prev->get_outputs(), 1, current->weights(), current->get_weights_lda());
    }
  }

  static void train_single(std::shared_ptr<NeuralNetworkMultilayerPerceptronCU<ActivationFunction> > neural_network, std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > labels, double eta =
      0.001) {
    neural_network->mse() = 0;
    for (std::size_t i = 0; i < labels.size(); i++) {
      train_single(neural_network, labels[i].first, labels[i].second, eta);
    }
    neural_network->mse() /= 2 * labels.size();
  }

  static void train(std::shared_ptr<NeuralNetworkMultilayerPerceptronCU<ActivationFunction> > neural_network, std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > labels
      , std::size_t max_rounds = 100 , double max_error = 0.001, double eta = 0.001) {
    for (std::size_t i = 0; i != max_rounds && neural_network->mse() > max_error; i++) {
      std::cout << "Backprop round " << i;
      train_single(neural_network, labels, eta);
      std::cout << " mse: " << neural_network->mse() << std::endl;
    }
  }
};

#endif /* BACKPROPAGATIONCU_HPP_ */
