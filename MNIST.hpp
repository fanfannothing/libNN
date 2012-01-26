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
  static std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > get_train();
  static std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > get_test();

  static std::size_t get_vector_size();
  static std::size_t get_output_size();

protected:
  static void load();
  static void get_train_next(boost::numeric::ublas::vector<double>& feature, boost::numeric::ublas::vector<double>& target);
  static void get_test_next(boost::numeric::ublas::vector<double>& feature, boost::numeric::ublas::vector<double>& target);

  static std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > m_train;
  static std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > m_test;
};

#endif /* MNIST_HPP_ */
