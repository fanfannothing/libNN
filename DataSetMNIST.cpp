/*
 * MNIST.cpp
 *
 *  Created on: Jan 22, 2012
 *      Author: wchan
 */

#include "DataSetMNIST.hpp"
#include <cstdint>
#include <arpa/inet.h>
#include <cassert>

std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > DataSetMNIST::m_train;
std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > DataSetMNIST::m_test;

void DataSetMNIST::load() {
  std::ifstream test_image("mnist/t10k-images-idx3-ubyte", std::ifstream::in | std::ifstream::binary);
  std::ifstream test_label("mnist/t10k-labels-idx1-ubyte", std::ifstream::in | std::ifstream::binary);
  std::ifstream train_image("mnist/train-images-idx3-ubyte", std::ifstream::in | std::ifstream::binary);
  std::ifstream train_label("mnist/train-labels-idx1-ubyte", std::ifstream::in | std::ifstream::binary);

  uint32_t buffer;

  // test for magic
  test_image.read(reinterpret_cast<char*>(&buffer), sizeof(uint32_t));
  assert(ntohl(buffer) == 2051);
  test_label.read(reinterpret_cast<char*>(&buffer), sizeof(uint32_t));
  assert(ntohl(buffer) == 2049);
  train_image.read(reinterpret_cast<char*>(&buffer), sizeof(uint32_t));
  assert(ntohl(buffer) == 2051);
  train_label.read(reinterpret_cast<char*>(&buffer), sizeof(uint32_t));
  assert(ntohl(buffer) == 2049);

  // should be 10k and 60k
  test_image.read(reinterpret_cast<char*>(&buffer), sizeof(uint32_t));
  assert(ntohl(buffer) == 10000);
  test_label.read(reinterpret_cast<char*>(&buffer), sizeof(uint32_t));
  assert(ntohl(buffer) == 10000);
  train_image.read(reinterpret_cast<char*>(&buffer), sizeof(uint32_t));
  assert(ntohl(buffer) == 60000);
  train_label.read(reinterpret_cast<char*>(&buffer), sizeof(uint32_t));
  assert(ntohl(buffer) == 60000);

  // next 4 entries should be 28
  test_image.read(reinterpret_cast<char*>(&buffer), sizeof(uint32_t));
  assert(ntohl(buffer) == 28);
  test_image.read(reinterpret_cast<char*>(&buffer), sizeof(uint32_t));
  assert(ntohl(buffer) == 28);

  train_image.read(reinterpret_cast<char*>(&buffer), sizeof(uint32_t));
  assert(ntohl(buffer) == 28);
  train_image.read(reinterpret_cast<char*>(&buffer), sizeof(uint32_t));
  assert(ntohl(buffer) == 28);

  m_train.clear();
  m_test.clear();

  while (train_image.good() && train_label.good()) {
    std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > pair;

    pair.first.resize(get_feature_size(), false);
    pair.second.resize(get_label_size(), false);
    std::fill(pair.second.begin(), pair.second.end(), -1);

    for (std::size_t i = 0; i < get_feature_size(); i++) {
      pair.first[i] = train_image.get() / 255.0 * 2.0 - 1.0;
    }
    pair.second[train_label.get()] = 1.0;

    m_train.push_back(pair);

    train_image.peek();
    train_label.peek();
  }

  while (test_image.good() && test_label.good()) {
    std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > pair;

    pair.first.resize(get_feature_size(), false);
    pair.second.resize(get_label_size(), false);
    std::fill(pair.second.begin(), pair.second.end(), -1);

    for (std::size_t i = 0; i < get_feature_size(); i++) {
      pair.first[i] = test_image.get() / 255.0 * 2.0 - 1.0;
    }
    pair.second[test_label.get()] = 1.0;

    m_test.push_back(pair);

    test_image.peek();
    test_label.peek();
  }

  // close our ifstream
  test_image.close();
  test_label.close();
  train_image.close();
  train_label.close();
}

std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > DataSetMNIST::get_train() {
  if (m_train.size() == 0) {
    load();
  }

  return m_train;
}

std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > DataSetMNIST::get_test() {
  if (m_test.size() == 0) {
    load();
  }
  return m_test;
}

std::size_t DataSetMNIST::get_feature_size() {
  return 28 * 28;
}
std::size_t DataSetMNIST::get_label_size() {
  return 10;
}
