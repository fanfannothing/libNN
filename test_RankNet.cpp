/*
 * test_RankNet.cpp
 *
 *  Created on: Feb 4, 2012
 *      Author: wchan
 */

#include "NeuralNetworkRankNet.hpp"
#include "DataSetLETOR.hpp"
#include "RankSet.hpp"

void test_RankNet() {
  RankSet train = DataSetLETOR::get_train();

  std::shared_ptr<NeuralNetworkRankNet> network(new NeuralNetworkRankNet(DataSetLETOR::get_feature_size(), 10));
}
