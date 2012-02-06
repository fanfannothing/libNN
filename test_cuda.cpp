/*
 * test_cuda.cpp
 *
 *  Created on: Jan 31, 2012
 *      Author: wchan
 */

#include "NeuralNetworkMultilayerPerceptron.hpp"
#include "NeuralNetworkMultilayerPerceptronCU.hpp"
#include "Backpropagation.hpp"
#include "BackpropagationCU.hpp"
#include "DataSetMNIST.hpp"
#include <ctime>

void test_mnist_test(std::shared_ptr<NeuralNetwork> network, std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > test) {
  int correct = 0;
  for (std::size_t i = 0; i < test.size(); i++) {
    boost::numeric::ublas::vector<double> output = network->f(test[i].first);

    if ((max_element(output.begin(), output.end()) - output.begin()) == (max_element(test[i].second.begin(), test[i].second.end()) - test[i].second.begin())) correct++;
  }
  std::cout << "correct " << correct << "/" << test.size() << " = " << ((double) correct / (double) test.size()) << std::endl;
}

void test_mnist(std::shared_ptr<NeuralNetwork> network, std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > train
    , std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > test) {

  test_mnist_test(network, train);
  test_mnist_test(network, test);
}

void test_mnist_test_cuda(std::shared_ptr<NeuralNetworkCU> network, std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > test) {
  int correct = 0;
  for (std::size_t i = 0; i < test.size(); i++) {
    boost::numeric::ublas::vector<double> output(10);
    output.clear();
    network->f(test[i].first, output);

    if ((max_element(output.begin(), output.end()) - output.begin()) == (max_element(test[i].second.begin(), test[i].second.end()) - test[i].second.begin())) correct++;
  }
  std::cout << "correct " << correct << "/" << test.size() << " = " << ((double) correct / (double) test.size()) << std::endl;
}

void test_mnist_cuda(std::shared_ptr<NeuralNetworkCU> network, std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > train
    , std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > test) {

  test_mnist_test_cuda(network, train);
  test_mnist_test_cuda(network, test);
}

void test_cuda() {
  assert(cublasInit() == CUBLAS_STATUS_SUCCESS);

  std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > train = DataSetMNIST::get_train();
  std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > test = DataSetMNIST::get_test();

  train.resize(100);

  std::vector<std::size_t> network_size = { DataSetMNIST::get_feature_size(), 300, DataSetMNIST::get_label_size() };
  std::shared_ptr<NeuralNetworkMultilayerPerceptron> original(new NeuralNetworkMultilayerPerceptron(network_size));
  std::shared_ptr<NeuralNetworkMultilayerPerceptronCU> network(new NeuralNetworkMultilayerPerceptronCU(original));

  double start = 0;
  double end = 0;

  start = std::clock();
  Backpropagation<>::train(original, train, 10, 0.001, 0.001);
  Backpropagation<>::train(original, train, 10, 0.001, 0.0001);
  end = std::clock();
  std::cout << (end - start) / 1000 << std::endl;

  start = std::clock();
  BackpropagationCU<>::train(network, train, 10, 0.001, 0.001);
  BackpropagationCU<>::train(network, train, 10, 0.001, 0.0001);
  end = std::clock();

  std::cout << (end - start) / 1000 << std::endl;

  test_mnist(original, train, test);
  test_mnist_cuda(network, train, test);

  cublasShutdown();
}

