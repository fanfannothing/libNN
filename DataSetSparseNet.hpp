/*
 * DataSetSparseNet.hpp
 *
 *  Created on: Feb 18, 2012
 *      Author: wchan
 */

#ifndef DATASETSPARSENET_HPP_
#define DATASETSPARSENET_HPP_

#include <iostream>
#include <fstream>
#include <boost/numeric/ublas/vector.hpp>

class DataSetSparseNet {
public:
  static std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > get_train();

  static std::size_t get_feature_size();
  static std::size_t get_label_size();

protected:
  static void load();
  static std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > m_train;
};

#endif /* DATASETSPARSENET_HPP_ */
