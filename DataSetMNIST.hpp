/*
 * MNIST.hpp
 *
 *  Created on: Jan 22, 2012
 *      Author: wchan
 */

#ifndef DATASETMNIST_HPP_
#define DATASETMNIST_HPP_

#include <iostream>
#include <fstream>
#include <boost/numeric/ublas/vector.hpp>

class DataSetMNIST {
public:
  static std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > get_train();
  static std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > get_test();

  static std::size_t get_feature_size();
  static std::size_t get_label_size();

protected:
  static void load();
  static std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > m_train;
  static std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > m_test;
};

#endif /* DATASETMNIST_HPP_ */
