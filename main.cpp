/*
 * main.cpp
 *
 *  Created on: Jan 21, 2012
 *      Author: wchan
 */

#include "ActivationFunctionTanh.hpp"
#include "ActivationFunctionSigmoid.hpp"
#include "NeuralNetworkMultiLayer.hpp"
#include "Backpropagation.hpp"
#include "ResilientBackpropagation.hpp"
#include "MNIST.hpp"

boost::numeric::ublas::vector<double> s0(1);
boost::numeric::ublas::vector<double> s1(1);
boost::numeric::ublas::vector<double> s00(2);
boost::numeric::ublas::vector<double> s01(2);
boost::numeric::ublas::vector<double> s10(2);
boost::numeric::ublas::vector<double> s11(2);
boost::numeric::ublas::vector<double> t0(1);
boost::numeric::ublas::vector<double> t1(1);
boost::numeric::ublas::vector<double> t00(2);
boost::numeric::ublas::vector<double> t01(2);
boost::numeric::ublas::vector<double> t10(2);
boost::numeric::ublas::vector<double> t11(2);

std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > s_xor;

void test_backpropagation_sigmoid() {
  std::vector<std::size_t> network_size = { 2, 3, 1 };
  std::shared_ptr<NeuralNetworkMultiLayer<ActivationFunctionSigmoid> > network(new NeuralNetworkMultiLayer<ActivationFunctionSigmoid>(network_size));

  for (std::size_t i = 0; i < 1000; i++) {
    network->mse() = 0;
    Backpropagation<ActivationFunctionSigmoid>::train(network, s00, s0);
    Backpropagation<ActivationFunctionSigmoid>::train(network, s01, s1);
    Backpropagation<ActivationFunctionSigmoid>::train(network, s10, s1);
    Backpropagation<ActivationFunctionSigmoid>::train(network, s11, s0);
    network->mse() /= 4;
  }

  assert(ActivationFunctionSigmoid::bound(network->f(s00)[0]) == s0[0]);
  assert(ActivationFunctionSigmoid::bound(network->f(s01)[0]) == s1[0]);
  assert(ActivationFunctionSigmoid::bound(network->f(s10)[0]) == s1[0]);
  assert(ActivationFunctionSigmoid::bound(network->f(s11)[0]) == s0[0]);
}

void test_backpropagation_tanh() {
  std::vector<std::size_t> network_size = { 2, 3, 1 };
  std::shared_ptr<NeuralNetworkMultiLayer<ActivationFunctionTanh> > network(new NeuralNetworkMultiLayer<ActivationFunctionTanh>(network_size));

  for (std::size_t i = 0; i < 5000; i++) {
    network->mse() = 0;
    Backpropagation<ActivationFunctionTanh>::train(network, t00, t0);
    Backpropagation<ActivationFunctionTanh>::train(network, t01, t1);
    Backpropagation<ActivationFunctionTanh>::train(network, t10, t1);
    Backpropagation<ActivationFunctionTanh>::train(network, t11, t0);
    network->mse() /= 4;
  }

  assert(ActivationFunctionTanh::bound(network->f(t00)[0]) == t0[0]);
  assert(ActivationFunctionTanh::bound(network->f(t01)[0]) == t1[0]);
  assert(ActivationFunctionTanh::bound(network->f(t10)[0]) == t1[0]);
  assert(ActivationFunctionTanh::bound(network->f(t11)[0]) == t0[0]);
}

void test_backpropagation() {
  test_backpropagation_sigmoid();
  //test_backpropagation_tanh();
}

void init_constants() {
  s0[0] = 0;
  s1[0] = 1;
  s00[0] = 0;
  s00[1] = 0;
  s01[0] = 0;
  s01[1] = 1;
  s10[0] = 1;
  s10[1] = 0;
  s11[0] = 1;
  s11[1] = 1;
  t0[0] = -1;
  t1[0] = 1;
  t00[0] = -1;
  t00[1] = -1;
  t01[0] = -1;
  t01[1] = 1;
  t10[0] = 1;
  t10[1] = -1;
  t11[0] = 1;
  t11[1] = 1;

  s_xor.push_back(std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> >(s00, s0));
  s_xor.push_back(std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> >(s01, s1));
  s_xor.push_back(std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> >(s10, s1));
  s_xor.push_back(std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> >(s11, s0));
}

void test_mnist() {
  std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > train;
  std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > test;

  MNIST::get_train(train);
  MNIST::get_test(test);

  std::vector<std::size_t> network_size = { MNIST::get_vector_size(), 64, MNIST::get_output_size() };
  std::shared_ptr<NeuralNetworkMultiLayer<ActivationFunctionSigmoid> > network(new NeuralNetworkMultiLayer<ActivationFunctionSigmoid>(network_size));

  train.resize(500);
  test.resize(100);

  for (std::size_t i = 0; i < 100; i++) {
    Backpropagation<ActivationFunctionSigmoid>::train(network, train);
  }

  int correct = 0;
  for (std::size_t i = 0; i < train.size(); i++) {
    boost::numeric::ublas::vector<double> output = network->f(train[i].first);

    if (index_norm_inf(output) == index_norm_inf(train[i].second)) correct++;
  }
  std::cout << "self correct " << (double) ((double) correct / (double) train.size()) << std::endl;

  correct = 0;
  for (std::size_t i = 0; i < test.size(); i++) {
    boost::numeric::ublas::vector<double> output = network->f(test[i].first);

    if (index_norm_inf(output) == index_norm_inf(test[i].second)) correct++;
  }
  std::cout << "test correct " << (double) ((double) correct / (double) test.size()) << std::endl;
}

void test_rprop_xor() {
  std::vector<std::size_t> network_size = { 2, 3, 1 };
  std::shared_ptr<NeuralNetworkMultiLayer<ActivationFunctionSigmoid> > network(new NeuralNetworkMultiLayer<ActivationFunctionSigmoid>(network_size));

  ResilientBackpropagation<ActivationFunctionSigmoid>::train(network, s_xor, 100);

  int correct = 0;
  for (std::size_t i = 0; i < s_xor.size(); i++) {
    boost::numeric::ublas::vector<double> output = network->f(s_xor[i].first);

    if (index_norm_inf(output) == index_norm_inf(s_xor[i].second)) correct++;
  }
  assert((correct / s_xor.size()) == 1);
}

void test_rprop_mnist() {
  std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > train;
  std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > test;

  MNIST::get_train(train);
  MNIST::get_test(test);



  std::vector<std::size_t> network_size = { MNIST::get_vector_size(), 300, MNIST::get_output_size() };
  std::shared_ptr<NeuralNetworkMultiLayer<ActivationFunctionSigmoid> > network(new NeuralNetworkMultiLayer<ActivationFunctionSigmoid>(network_size));

  ResilientBackpropagation<ActivationFunctionSigmoid>::train(network, train, 100);

  int correct = 0;
  for (std::size_t i = 0; i < train.size(); i++) {
    boost::numeric::ublas::vector<double> output = network->f(train[i].first);

    if (index_norm_inf(output) == index_norm_inf(train[i].second)) correct++;

  }
  std::cout << "self correct " << correct << "/" << train.size() << " = " << ((double) correct / (double) train.size()) << std::endl;

  correct = 0;
  for (std::size_t i = 0; i < test.size(); i++) {
    boost::numeric::ublas::vector<double> output = network->f(test[i].first);

    if (index_norm_inf(output) == index_norm_inf(test[i].second)) correct++;
  }
  std::cout << "test correct " << correct << "/" << test.size() << " = " << ((double) correct / (double) test.size()) << std::endl;
}

int main(int argc, char* argv[]) {
  std::cout.setf(std::ios_base::fixed);
  std::cout.precision(4);

  init_constants();

  test_backpropagation();
  //test_mnist();
  test_rprop_xor();
  test_rprop_mnist();

  return 1;
}