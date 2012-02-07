/*
 * NeuralNetworkRankNet.hpp
 *
 *  Created on: Feb 4, 2012
 *      Author: wchan
 */

#ifndef NEURALNETWORKRANKNET_HPP_
#define NEURALNETWORKRANKNET_HPP_

#include "ActivationFunctionLinear.hpp"
#include "NeuralNetworkLayer.hpp"
#include "NeuralNetworkMultilayerPerceptron.hpp"
#include <boost/numeric/ublas/vector.hpp>
#include "LossFunctionRankNet.hpp"
#include "RankSet.hpp"

/**
 * Structure is one layer of hidden tanh nodes, followed a linear node
 */
class NeuralNetworkRankNet {
public:
  NeuralNetworkRankNet(std::size_t input, std::size_t hidden) {
    m_network_0.reset(new NeuralNetworkMultilayerPerceptron( { input, hidden }));
    std::shared_ptr<NeuralNetworkLayer> linear(new NeuralNetworkLayer(1, m_network_0->get_layers()[m_network_0->get_layers().size() - 1], std::shared_ptr<ActivationFunction>(new ActivationFunctionLinear())));
    m_network_0->add_layer(linear);
    m_network_1.reset(m_network_0->clone());

    m_eta = 0.001;
  }
  virtual ~NeuralNetworkRankNet() {
  }

  virtual boost::numeric::ublas::vector<double> f0(boost::numeric::ublas::vector<double> in) {
    return m_network_0->f(in);
  }

  virtual boost::numeric::ublas::vector<double> f1(boost::numeric::ublas::vector<double> in) {
    return m_network_1->f(in);
  }

  /**
   * Performs a pair-wise...
   */
  virtual double test_pair(RankSet& set) {
    double count = 0;
    double sum = 0;
    for (std::unordered_map<std::size_t, RankList>::iterator it = set.get_map().begin(); it != set.get_map().end(); it++) {
      sum += test_pair(it->second);
      count += 1;
    }
    return sum / count;
  }

  virtual double test_pair(RankList& list) {
    double correct = 0;
    double count = 0;

    for (std::size_t i = 0; i < list.get_list().size(); i++) {
      for (std::size_t j = i + 1; j < list.get_list().size(); j++) {
        double oi = f0(list.get_list()[i].first)[0];
        double oj = f0(list.get_list()[j].first)[0];

        if (((oi - oj) * (list.get_list()[i].second - list.get_list()[j].second)) >= 0) correct++;
        count++;
      }
    }

    return correct / count;
  }

  virtual void train(RankSet& set) {
    for (std::unordered_map<std::size_t, RankList>::iterator it = set.get_map().begin(); it != set.get_map().end(); it++) {
      train(it->second);
    }
  }

  virtual void train(RankList& list) {
    for (std::size_t i = 0; i < list.get_list().size(); i++) {
      for (std::size_t j = i + 1; j < list.get_list().size(); j++) {
        train(list.get_list()[i].first, list.get_list()[j].first);
      }
    }
  }

  virtual void rank(RankSet& set) {
    for (std::unordered_map<std::size_t, RankList>::iterator it = set.get_map().begin(); it != set.get_map().end(); it++) {
      rank(it->second);
    }
  }

  virtual void rank(RankList& list) {
    for (std::size_t i = 0; i < list.get_list().size(); i++) {
      double r = m_network_0->f(list.get_list()[i].first)[0];
      list.get_list()[i].third = r;
    }

    list.sort_ranked();
  }

  /**
   * This function has substantial similiarities to Backpropagation... however there are subtle differences so it has been decided to not use the Backpropagation class due to efficiecy reasons.
   */
  virtual void train(boost::numeric::ublas::vector<double> x0, boost::numeric::ublas::vector<double> x1) {
    // forward pass...
    double oi = m_network_0->f(x0)[0];
    double oj = m_network_1->f(x1)[0];

    // probabilistic cost function
    double pij = 1.0 / (1.0 + std::exp(-(oi - oj)));

    std::vector<std::shared_ptr<NeuralNetworkLayer> > layers_0 = m_network_0->get_layers();
    std::vector<std::shared_ptr<NeuralNetworkLayer> > layers_1 = m_network_1->get_layers();

    // set our dedx now...
    std::shared_ptr<NeuralNetworkLayer> last_layer_0 = m_network_0->get_layers()[m_network_0->get_layers().size() - 1];
    std::shared_ptr<NeuralNetworkLayer> last_layer_1 = m_network_1->get_layers()[m_network_1->get_layers().size() - 1];

    last_layer_0->dedx() = (1 - pij) * last_layer_0->dydx();
    last_layer_1->dedx() = (1 - pij) * last_layer_1->dydx();

    // now it's time to backpropgate our dedy...
    for (std::size_t i = layers_0.size() - 2; i <= layers_0.size(); i--) {
      std::shared_ptr<NeuralNetwork> current_0 = layers_0[i];
      std::shared_ptr<NeuralNetwork> next_0 = layers_0[i + 1];

      boost::numeric::ublas::vector<double> current_dedx_0 = prod(next_0->dedx(), next_0->weights());
      std::transform(current_dedx_0.begin(), current_dedx_0.end(), current_0->dydx().begin(), current_dedx_0.begin(), std::multiplies<double>());

      current_0->dedx() = current_dedx_0;

      std::shared_ptr<NeuralNetwork> current_1 = layers_1[i];
      std::shared_ptr<NeuralNetwork> next_1 = layers_1[i + 1];

      boost::numeric::ublas::vector<double> current_dedx_1 = prod(next_1->dedx(), next_1->weights());
      std::transform(current_dedx_1.begin(), current_dedx_1.end(), current_1->dydx().begin(), current_dedx_1.begin(), std::multiplies<double>());

      current_1->dedx() = current_dedx_1;
    }

    // now to update our weights...
    for (std::size_t i = 0; i < layers_0.size(); i++) {
      std::shared_ptr<NeuralNetwork> current_0 = layers_0[i];
      std::shared_ptr<NeuralNetwork> prev_0 = current_0->get_inputs();
      std::shared_ptr<NeuralNetwork> current_1 = layers_1[i];
      std::shared_ptr<NeuralNetwork> prev_1 = current_1->get_inputs();

      boost::numeric::ublas::matrix<double> delta = m_eta * (outer_prod(current_0->dedx(), prev_0->get_outputs()) - outer_prod(current_1->dedx(), prev_1->get_outputs()));

      // updating the weights twice is probably faster than doing a copy/clone later?
      current_0->weights() += delta;
      current_1->weights() += delta;
    }
  }

  virtual void train() {

  }

protected:
  double m_eta;

  std::shared_ptr<NeuralNetworkMultilayerPerceptron> m_network_0;
  std::shared_ptr<NeuralNetworkMultilayerPerceptron> m_network_1;
}
;

#endif /* NEURALNETWORKRANKNET_HPP_ */
