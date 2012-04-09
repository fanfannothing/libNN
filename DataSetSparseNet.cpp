/*
 * DataSetSparseNet.cpp
 *
 *  Created on: Feb 18, 2012
 *      Author: wchan
 */

#include "DataSetSparseNet.hpp"

#include <cstdint>
#include <cassert>
#include <matio.h>

std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > DataSetMNIST::m_train;

void DataSetSparseNet::load() {
  mat_t train = Mat_Open("sparsenet/IMAGES.mat", MAT_ACC_RDONLY);

  Mat_Close(train);
}

