/*
 * LETOR.hpp
 *
 *  Created on: Feb 4, 2012
 *      Author: wchan
 */

#ifndef DATASETLETOR_HPP_
#define DATASETLETOR_HPP_

#include "RankSet.hpp"
#include <boost/numeric/ublas/vector.hpp>

class DataSetLETOR {
public:
  static RankSet get_train();
  static RankSet get_test();

  static std::size_t get_feature_size();
  static std::size_t get_label_size();

protected:
  static void load();
  static RankSet m_train;
  static RankSet m_test;
};

#endif /* DATASETLETOR_HPP_ */
