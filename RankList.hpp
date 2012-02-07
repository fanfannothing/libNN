/*
 * RankList.hpp
 *
 *  Created on: Feb 6, 2012
 *      Author: wchan
 */

#ifndef RANKLIST_HPP_
#define RANKLIST_HPP_

#include <boost/numeric/ublas/vector.hpp>

template<class T, class U, class V>
struct triplet {
public:
  triplet() {
  }
  triplet(T t, U u, V v) {
    first = t;
    second = u;
    third = v;
  }

  T first;
  U second;
  V third;
};

typedef triplet<boost::numeric::ublas::vector<double>, double, double> RankListEntry;

/**
 * This class is in support for the RankNet (or ranking algorithms in general)
 *
 * It is basically a simple wrapper... we could do without it but it would mean very long lines of code of template wrapping
 */
class RankList {
public:
  struct RankListComparator2 {
    bool operator()(const RankListEntry& a, const RankListEntry& b) {
      return a.second > b.second;
    }
  };

  struct RankListComparator3 {
    bool operator()(const RankListEntry& a, const RankListEntry& b) {
      return a.third > b.third;
    }
  };

  void push_back(boost::numeric::ublas::vector<double> features, double label) {
    m_list.push_back(RankListEntry(features, label, 0));
  }

  void sort_truth() {
    std::sort(m_list.begin(), m_list.end(), RankListComparator2());
    m_normalization = 1 / get_discounted_cumulative_gain();
  }

  void sort_ranked() {
    std::sort(m_list.begin(), m_list.end(), RankListComparator3());
  }

  std::vector<RankListEntry>& get_list() {
    return m_list;
  }

  /* assumes list has been sorted properly and takes the DCG in the order as is */
  double get_discounted_cumulative_gain() {
    double dcg = 0;
    for (std::size_t i = 0; i < m_list.size(); i++) {
      dcg += (std::pow(2, m_list[i].second) - 1) / (std::log(2 + i));
    }
    return dcg;
  }

  double get_normalized_discounted_cmulative_gain() {
    return m_normalization * get_discounted_cumulative_gain();
  }

  void print() {
    for (std::size_t i = 0; i < m_list.size(); i++) {
      std::cout << m_list[i].second << std::endl;
    }
  }

protected:
  std::vector<RankListEntry> m_list;
  double m_normalization;
};

#endif /* RANKLIST_HPP_ */
