/*
 * test_LambdaRank.cpp
 *
 *  Created on: Feb 11, 2012
 *      Author: wchan
 */

#include "NeuralNetworkLambdaRank.hpp"
#include "DataSetLETOR.hpp"
#include "RankSet.hpp"

void test_LambdaRank() {
  std::cout << "test_LambdaRank" << std::endl;
  RankSet train = DataSetLETOR::get_train();
  RankSet test = DataSetLETOR::get_test();

  std::shared_ptr<NeuralNetworkLambdaRank> network(new NeuralNetworkLambdaRank(DataSetLETOR::get_feature_size(), 10));

  for (int i = 0; i < 50; i++)
    network->train(train);

  network->rank(train);
  network->rank(test);

  std::unordered_map<std::size_t, RankList> map = test.get_map();

  double rough = 0;
  double count = 0;
  for (std::unordered_map<std::size_t, RankList>::iterator it = map.begin(); it != map.end(); it++) {
    rough += it->second.get_normalized_discounted_cumulative_gain();
    count++;
  }

  std::cout << "rough average ncdg: " << rough / count << std::endl;
}

