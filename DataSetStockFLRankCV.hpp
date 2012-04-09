/*
 * DataSetStockFLRankCV.hpp
 *
 *  Created on: Mar 16, 2012
 *      Author: wchan
 */

#ifndef DATASETSTOCKFLRANKCV_HPP_
#define DATASETSTOCKFLRANKCV_HPP_

#include "RankSet.hpp"
#include <boost/numeric/ublas/vector.hpp>

class DataSetStockFLRankCV {
public:
  static RankSet get_train(std::size_t cv, int mod);
  static std::vector<boost::numeric::ublas::vector<double> > get_test(std::size_t cv, int mod);

  static std::size_t get_feature_size();
  static std::size_t get_label_size();
};

#endif
