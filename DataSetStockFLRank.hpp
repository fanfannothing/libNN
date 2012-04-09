/*
 * DataSetStockFLRank.hpp
 *
 *  Created on: Mar 15, 2012
 *      Author: wchan
 */

#ifndef DATASETSTOCKFLRANK_HPP_
#define DATASETSTOCKFLRANK_HPP_

#include "RankSet.hpp"
#include <boost/numeric/ublas/vector.hpp>

class DataSetStockFLRank {
public:
  static RankSet get_train();
  static RankSet get_test();

  static std::size_t get_feature_size();
  static std::size_t get_label_size();

  static std::vector<boost::numeric::ublas::vector<double> > m_test_unsorted;
protected:
  static void load();
  static RankSet m_train;
  static RankSet m_test;
};

#endif /* DATASETSTOCKFLRANK_HPP_ */
