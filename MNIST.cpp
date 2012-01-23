/*
 * MNIST.cpp
 *
 *  Created on: Jan 22, 2012
 *      Author: wchan
 */

#include "MNIST.hpp"
#include <cstdint>
#include <arpa/inet.h>
#include <cassert>

std::ifstream MNIST::m_test_image;
std::ifstream MNIST::m_test_label;
std::ifstream MNIST::m_train_image;
std::ifstream MNIST::m_train_label;

uint32_t MNIST::m_vector_size = 28 * 28;
uint32_t MNIST::m_output_size = 10;

void MNIST::load() {
  const std::string test_image = "mnist/t10k-images-idx3-ubyte";
  const std::string test_label = "mnist/t10k-labels-idx1-ubyte";
  const std::string train_image = "mnist/train-images-idx3-ubyte";
  const std::string train_label = "mnist/train-labels-idx1-ubyte";

  m_test_image.open(test_image, std::ifstream::in | std::ifstream::binary);
  m_test_label.open(test_label, std::ifstream::in | std::ifstream::binary);
  m_train_image.open(train_image, std::ifstream::in | std::ifstream::binary);
  m_train_label.open(train_label, std::ifstream::in | std::ifstream::binary);

  uint32_t test_image_magic = 0;
  uint32_t test_label_magic = 0;
  uint32_t train_image_magic = 0;
  uint32_t train_label_magic = 0;

  m_test_image.read(reinterpret_cast<char*>(&test_image_magic), sizeof(uint32_t));
  m_test_label.read(reinterpret_cast<char*>(&test_label_magic), sizeof(uint32_t));
  m_train_image.read(reinterpret_cast<char*>(&train_image_magic), sizeof(uint32_t));
  m_train_label.read(reinterpret_cast<char*>(&train_label_magic), sizeof(uint32_t));

  assert(ntohl(test_image_magic) == 2051);
  assert(ntohl(test_label_magic) == 2049);
  assert(ntohl(train_image_magic) == 2051);
  assert(ntohl(train_label_magic) == 2049);

  uint32_t buffer;

  // should be 10k and 60k
  m_test_image.read(reinterpret_cast<char*>(&buffer), sizeof(uint32_t));
  assert(ntohl(buffer) == 10000);
  m_test_label.read(reinterpret_cast<char*>(&buffer), sizeof(uint32_t));
  assert(ntohl(buffer) == 10000);
  m_train_image.read(reinterpret_cast<char*>(&buffer), sizeof(uint32_t));
  assert(ntohl(buffer) == 60000);
  m_train_label.read(reinterpret_cast<char*>(&buffer), sizeof(uint32_t));
  assert(ntohl(buffer) == 60000);

  // next 4 entries should be 28
  m_test_image.read(reinterpret_cast<char*>(&buffer), sizeof(uint32_t));
  assert(ntohl(buffer) == 28);
  m_test_image.read(reinterpret_cast<char*>(&buffer), sizeof(uint32_t));
  assert(ntohl(buffer) == 28);

  m_train_image.read(reinterpret_cast<char*>(&buffer), sizeof(uint32_t));
  assert(ntohl(buffer) == 28);
  m_train_image.read(reinterpret_cast<char*>(&buffer), sizeof(uint32_t));
  assert(ntohl(buffer) == 28);
}

void MNIST::close() {
  m_test_image.close();
  m_test_label.close();
  m_train_image.close();
  m_train_label.close();
}

bool MNIST::has_train_next() {
  m_train_image.peek();
  m_train_label.peek();
  return m_train_image.good() && m_train_label.good();
}

bool MNIST::has_test_next() {
  m_test_image.peek();
  m_test_label.peek();
  return m_test_image.good() && m_test_label.good();
}

void MNIST::get_train_next(boost::numeric::ublas::vector<double>& feature, boost::numeric::ublas::vector<double>& target) {
  feature.resize(m_vector_size, false);
  target.resize(10, false);
  target.clear();

  for (std::size_t i = 0; i < m_vector_size; i++) {
    feature[i] = m_train_image.get() / 255.0;
  }
  target[m_train_label.get()] = 1.0;
}

void MNIST::get_test_next(boost::numeric::ublas::vector<double>& feature, boost::numeric::ublas::vector<double>& target) {
  feature.resize(m_vector_size, false);
  target.resize(10, false);
  target.clear();

  for (std::size_t i = 0; i < m_vector_size; i++) {
    feature[i] = m_test_image.get() / 255.0;
  }

  target[m_test_label.get()] = 1.0;
}

std::size_t MNIST::get_vector_size() {
  return m_vector_size;
}

std::size_t MNIST::get_output_size() {
  return m_output_size;
}

void MNIST::get_train(std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > >& set) {
  set.clear();
  load();

  while (has_train_next()) {
    std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > pair;
    get_train_next(pair.first, pair.second);
    set.push_back(pair);
  }

  close();
  std::cerr << "MNIST::get_train :: done." << std::endl;
}

void MNIST::get_test(std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > >& set) {
  set.clear();
  load();

  while (has_test_next()) {
    std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > pair;
    get_test_next(pair.first, pair.second);
    set.push_back(pair);
  }

  close();
  std::cerr << "MNIST::get_test :: done." << std::endl;
}
