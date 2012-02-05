/*
 * LETOR.cpp
 *
 *  Created on: Feb 4, 2012
 *      Author: wchan
 */

#include "LETOR.hpp"

std::size_t LETOR::get_vector_size() {
  return 136;
}

std::size_t LETOR::get_output_size() {
  return 1;
}

std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > LETOR::get_train() {
  if (m_train.size() == 0) {
    load();
  }

  return m_train;
}

std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > LETOR::get_test() {
  if (m_test.size() == 0) {
    load();
  }
  return m_test;
}
