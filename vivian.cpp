/*
 * vivian.cpp
 *
 *  Created on: Feb 17, 2012
 *      Author: wchan
 */

#include "NeuralNetworkMultilayerPerceptron.hpp"
#include "NeuralNetworkMultilayerPerceptronCU.hpp"
#include "NeuralNetworkVivian.hpp"
#include "Backpropagation.hpp"
#include "BackpropagationCU.hpp"
#include "DataSetMNIST.hpp"

void test_mnist(std::shared_ptr<NeuralNetwork> network, std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > train
    , std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > test);

void test_mnist_cuda(std::shared_ptr<NeuralNetworkCU> network, std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > train
    , std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > test);

void vivian() {
  std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > train = DataSetMNIST::get_train();
  std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > test = DataSetMNIST::get_test();

  train.resize(1000);

  std::vector<std::size_t> network_size = { DataSetMNIST::get_feature_size(), 300, DataSetMNIST::get_label_size() };
  std::shared_ptr<NeuralNetworkMultilayerPerceptronCU> original(new NeuralNetworkMultilayerPerceptronCU(network_size));

  BackpropagationCU<>::train(original, train, 10, 0.001, 0.001);
  BackpropagationCU<>::train(original, train, 10, 0.001, 0.0001);
  // BackpropagationCU<>::train(original, train, 10, 0.001, 0.00001);
  test_mnist_cuda(original, train, test);
  // correct 9688/10000 = 0.968800
  // correct 9435/10000 = 0.943500

  // full set with 10+10+10 rounds of training
  // correct 59317/60000 = 0.988617
  // correct 9782/10000 = 0.978200

  /*
  std::cout << "vivian " << std::endl;
  std::shared_ptr<NeuralNetworkVivian> v(new NeuralNetworkVivian());
  v->train();

  test_mnist(v, train, test);
  */

  // 10000 training exampls, 10+10 rounds of training
  // correct 9669/10000 = 0.966900
  // correct 9412/10000 = 0.941200

  // using linear net MSE
  // correct 9772/10000 = 0.977200
  // correct 9532/10000 = 0.953200

  // full set, 10+10+10 rounds of training
  // correct 59038/60000 = 0.983967
  // correct 9753/10000 = 0.975300

}
