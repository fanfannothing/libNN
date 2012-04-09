/*
 * NeuralNetworkVivian.hpp
 *
 *  Created on: Feb 17, 2012
 *      Author: wchan
 */

#ifndef NEURALNETWORKVIVIAN_HPP_
#define NEURALNETWORKVIVIAN_HPP_

#include "NeuralNetworkMultilayerPerceptron.hpp"
#include "NeuralNetworkFunctionApproximator.hpp"
#include "Backpropagation.hpp"
#include "DataSetMNIST.hpp"

class NeuralNetworkVivian : public NeuralNetwork {
public:
  NeuralNetworkVivian() {
    for (std::size_t i = 0; i < DataSetMNIST::get_label_size(); i++) {
      m_layer_shallow.push_back(std::shared_ptr<NeuralNetworkFunctionApproximator>(new NeuralNetworkFunctionApproximator(DataSetMNIST::get_feature_size(), 300, 1)));
    }

    m_layer_deep.reset(new NeuralNetworkMultilayerPerceptron( { DataSetMNIST::get_label_size(), 10, DataSetMNIST::get_label_size() }));
  }

  void train() {
    train_layer_shallow();
    train_layer_deep();
  }

  void train_layer_shallow() {
    std::cout << "train_layer_shallow" << std::endl;
    std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > train = DataSetMNIST::get_train();
    std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > train2;

    // train.resize(10000);

    for (std::size_t i = 0; i < m_layer_shallow.size(); i++) {
      train2.resize(0);

      for (std::size_t j = 0; j < train.size(); j++) {
        boost::numeric::ublas::vector<double> label(1);

        label[0] = train[j].second[i];

        train2.push_back(std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> >(train[j].first, label));
      }

      Backpropagation<>::train(m_layer_shallow[i], train2, 10, 0.001, 0.001);
      Backpropagation<>::train(m_layer_shallow[i], train2, 10, 0.001, 0.0001);
      Backpropagation<>::train(m_layer_shallow[i], train2, 10, 0.001, 0.00001);
    }
  }

  void train_layer_deep() {
    std::cout << "train_deep" << std::endl;
    std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > train = DataSetMNIST::get_train();
    std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > train2;

    // train.resize(10000);

    for (std::size_t i = 0; i < train.size(); i++) {
      // assert(m_layer_shallow.size() == DataSetMNIST::get_label_size());

      boost::numeric::ublas::vector<double> feature(DataSetMNIST::get_label_size());

      for (std::size_t j = 0; j < m_layer_shallow.size(); j++) {
        feature[j] = m_layer_shallow[j]->f(train[i].first)[0];
      }

      train2.push_back(std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> >(feature, train[i].second));
    }

    Backpropagation<>::train(m_layer_deep, train2, 10, 0.001, 0.001);
    Backpropagation<>::train(m_layer_deep, train2, 10, 0.001, 0.0001);
    Backpropagation<>::train(m_layer_deep, train2, 10, 0.001, 0.00001);
  }

  virtual boost::numeric::ublas::vector<double> f(boost::numeric::ublas::vector<double> x) {
    boost::numeric::ublas::vector<double> feature(m_layer_shallow.size());

    for (std::size_t i = 0; i < m_layer_shallow.size(); i++) {
      feature[i] = m_layer_shallow[i]->f(x)[0];
    }

    return m_layer_deep->f(feature);
  }

  virtual void compute() {
    throw;
  }

  virtual std::size_t get_outputs_size() {
    return DataSetMNIST::get_label_size();
  }

protected:
  std::vector<std::shared_ptr<NeuralNetworkMultilayerPerceptron> > m_layer_shallow;
  std::shared_ptr<NeuralNetworkMultilayerPerceptron> m_layer_deep;
};

#endif /* NEURALNETWORKVIVIAN_HPP_ */
