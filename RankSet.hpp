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
#include <limits>

class RankSet {
public:
  void push_back(std::size_t query_id, boost::numeric::ublas::vector<double> features, double label) {
    m_set[query_id].push_back(features, label);
  }

  void set_ranklist(std::size_t query_id, const RankList& list) {
    m_set[query_id] = list;
  }

  std::size_t size() {
    return m_set.size();
  }

  void sort_truth() {
    for (std::unordered_map<std::size_t, RankList>::iterator it = m_set.begin(); it != m_set.end(); it++) {
      it->second.sort_truth();

      if (it->second.get_reciprical_max_discounted_cumulative_gain() == std::numeric_limits<double>::infinity()) m_set.erase(it);
    }
  }

  void print() {
    for (std::unordered_map<std::size_t, RankList>::iterator it = m_set.begin(); it != m_set.end(); it++) {
      std::cout << it->first << " " << it->second.get_list().size() << std::endl;
    }
  }

  void clear() {
    m_set.clear();
  }

  std::unordered_map<std::size_t, RankList>& get_map() {
    return m_set;
  }

protected:
  std::unordered_map<std::size_t, RankList> m_set;
};

#endif /* RANKSET_HPP_ */
