/*
 * RankList.hpp
 *
 *  Created on: Feb 6, 2012
 *      Author: wchan
 */

#ifndef RANKLIST_HPP_
#define RANKLIST_HPP_

#include <boost/numeric/ublas/vector.hpp>

/**
 * This class is in support for the RankNet (or ranking algorithms in general)
 *
 * It is basically a simple wrapper... we could do without it but it would mean very long lines of code of template wrapping
 */
class RankList {
public:
  static bool Comparator(std::pair<boost::numeric::ublas::vector<double>, double> a, std::pair<boost::numeric::ublas::vector<double>, double> b) {
    return a.second >= b.second;
  }

  void push_back(boost::numeric::ublas::vector<double> features, double label) {
    m_list.push_back(std::pair<boost::numeric::ublas::vector<double>, double>(features, label));
  }

  void sort() {
    std::sort(m_list.begin(), m_list.end(), Comparator);
  }

  std::vector<std::pair<boost::numeric::ublas::vector<double>, double> > get_list() const {
    return m_list;
  }

protected:
  std::vector<std::pair<boost::numeric::ublas::vector<double>, double> > m_list;
};

#endif /* RANKLIST_HPP_ */
