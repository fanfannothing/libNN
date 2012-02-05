/*
 * LETOR.hpp
 *
 *  Created on: Feb 4, 2012
 *      Author: wchan
 */

#ifndef LETOR_HPP_
#define LETOR_HPP_

#include <boost/numeric/ublas/vector.hpp>

class LETOR {
public:
  static std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > get_train();
  static std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > get_test();

  static std::size_t get_vector_size();
  static std::size_t get_output_size();

protected:
  static void load();
  static std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > m_train;
  static std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > m_test;
};

#endif /* LETOR_HPP_ */
