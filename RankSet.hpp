/*
 * RankSet.hpp
 *
 *  Created on: Feb 6, 2012
 *      Author: wchan
 */

#ifndef RANKSET_HPP_
#define RANKSET_HPP_

#include "RankList.hpp"
#include <unordered_map>

class RankSet {
public:
  void push_back(std::size_t query_id, boost::numeric::ublas::vector<double> features, double label) {
    m_set[query_id].push_back(features, label);
  }

  std::size_t size() {
    return m_set.size();
  }

  void clear() {
    m_set.clear();
  }

protected:
  std::unordered_map<std::size_t, RankList> m_set;
};

#endif /* RANKSET_HPP_ */
