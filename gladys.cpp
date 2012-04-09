/*
 * gladys.cpp
 *
 *  Created on: Feb 19, 2012
 *      Author: wchan
 */

#include "NeuralNetworkGladys.hpp"

void test_mnist(std::shared_ptr<NeuralNetwork> network, std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > train
    , std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > test);

void gladys() {
  std::shared_ptr<NeuralNetworkGladys> g(new NeuralNetworkGladys());
  g->train();
  // g->test2();

  std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > train = DataSetMNIST::get_train();
  std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > test = DataSetMNIST::get_test();

  //train.resize(10000);

  test_mnist(g, train, test);
}
