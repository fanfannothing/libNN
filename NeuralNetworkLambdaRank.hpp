/*
 * NeuralNetworkLambdaRank.hpp
 *
 *  Created on: Feb 8, 2012
 *      Author: wchan
 */

#ifndef NEURALNETWORKLAMBDARANK_HPP_
#define NEURALNETWORKLAMBDARANK_HPP_

#include "ActivationFunctionLinear.hpp"
#include "NeuralNetworkLayer.hpp"
#include "NeuralNetworkMultilayerPerceptron.hpp"
#include <boost/numeric/ublas/vector.hpp>
#include "RankSet.hpp"

/**
 * LambdaRank
 *
 * Learning to Rank with Nonsmooth Cost Functions
 */
class NeuralNetworkLambdaRank {
public:
  NeuralNetworkLambdaRank(std::size_t input, std::size_t hidden) {
    m_network.reset(new NeuralNetworkMultilayerPerceptron( { input, hidden }));
    std::shared_ptr<NeuralNetworkLayer> linear(new NeuralNetworkLayer(1, m_network->get_layers()[m_network->get_layers().size() - 1], std::shared_ptr<ActivationFunction>(new ActivationFunctionLinear())));
    m_network->add_layer(linear);

    m_eta = 0.0001;
  }

  virtual ~NeuralNetworkLambdaRank() {
  }

  double lambda_function(std::size_t i, std::size_t j, double si, double sj, double ri, double rj, double normization) {
    return normization * (1.0 / (1.0 + std::exp(si - sj))) * (std::pow(2.0, ri) - std::pow(2.0, rj)) * (std::log((2.0 + j) / (2.0 + i)));
  }

  virtual void train(RankSet& set) {
    for (std::unordered_map<std::size_t, RankList>::iterator it = set.get_map().begin(); it != set.get_map().end(); it++) {
      train(it->second);
    }
  }

  virtual void train(RankList& list) {
    m_output_cache.resize(list.get_list().size());

    // calculate outputs and cache them
    for (std::size_t j = 0; j < list.get_list().size(); j++) {
      m_output_cache[j] = m_network->f(list.get_list()[j].first)[0];
    }

    for (std::size_t i = 0; i < list.get_list().size(); i++) {
      // calculate lambda
      double lambda = 0;
      for (std::size_t j = 0; j < list.get_list().size(); j++) {
        // if (i != j)
        lambda += lambda_function(i, j, m_output_cache[i], m_output_cache[j], list.get_list()[i].second, list.get_list()[j].second, list.get_reciprical_max_discounted_cumulative_gain());
      }

      if (lambda != lambda) {
        throw;
      }

      train(list.get_list()[i].first, lambda);
    }
  }

  virtual void rank(RankSet& set) {
    for (std::unordered_map<std::size_t, RankList>::iterator it = set.get_map().begin(); it != set.get_map().end(); it++) {
      rank(it->second);
    }
  }

  virtual void rank(RankList& list) {
    for (std::size_t i = 0; i < list.get_list().size(); i++) {
      double r = m_network->f(list.get_list()[i].first)[0];
      list.get_list()[i].third = r;
    }

    list.sort_ranked();
  }

  virtual void train(boost::numeric::ublas::vector<double> x, double lambda) {
    // the gradient is defined by the error function...
    // double lambda = lambda_function();

    // gotta rerun it... we can optimize this operation cache later?
    m_network->f(x);

    std::vector<std::shared_ptr<NeuralNetworkLayer> > layers = m_network->get_layers();

    // set our dedx now...
    std::shared_ptr<NeuralNetworkLayer> last_layer = m_network->get_layers()[m_network->get_layers().size() - 1];

    last_layer->dedx() = -lambda * last_layer->dydx();

    // now it's time to backpropgate our dedy...
    for (std::size_t i = layers.size() - 2; i <= layers.size(); i--) {
      std::shared_ptr<NeuralNetwork> current = layers[i];
      std::shared_ptr<NeuralNetwork> next = layers[i + 1];

      boost::numeric::ublas::vector<double> current_dedx = prod(next->dedx(), next->weights());
      std::transform(current_dedx.begin(), current_dedx.end(), current->dydx().begin(), current_dedx.begin(), std::multiplies<double>());

      current->dedx() = current_dedx;
    }

    // now to update our weights...
    for (std::size_t i = 0; i < layers.size(); i++) {
      std::shared_ptr<NeuralNetwork> current = layers[i];
      std::shared_ptr<NeuralNetwork> prev = current->get_inputs();

      current->weights() += m_eta * (outer_prod(current->dedx(), prev->get_outputs()));
    }
  }

protected:
  double m_eta;

  std::shared_ptr<NeuralNetworkMultilayerPerceptron> m_network;
  boost::numeric::ublas::vector<double> m_output_cache;
};

#endif /* NEURALNETWORKLAMBDARANK_HPP_ */
