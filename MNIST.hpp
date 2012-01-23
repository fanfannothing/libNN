/*
 * MNIST.hpp
 *
 *  Created on: Jan 22, 2012
 *      Author: wchan
 */

#ifndef MNIST_HPP_
#define MNIST_HPP_

#include <iostream>
#include <fstream>
#include <boost/numeric/ublas/vector.hpp>

class MNIST {
public:
  static void load();
  static void close();
  static void get_train_next(boost::numeric::ublas::vector<double>& feature, boost::numeric::ublas::vector<double>& target);
  static void get_test_next(boost::numeric::ublas::vector<double>& feature, boost::numeric::ublas::vector<double>& target);

  static bool has_train_next();
  static bool has_test_next();

  static void get_train(std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > >& set);
  static void get_test(std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > >& set);

  static std::size_t get_vector_size();
  static std::size_t get_output_size();

protected:
  static std::ifstream m_test_image;
  static std::ifstream m_test_label;
  static std::ifstream m_train_image;
  static std::ifstream m_train_label;

  static uint32_t m_vector_size;
  static uint32_t m_output_size;
};

#endif /* MNIST_HPP_ */
