/*
 * test_stock.cpp
 *
 *  Created on: Mar 15, 2012
 *      Author: wchan
 */

#include "NeuralNetworkRankNet.hpp"
#include "DataSetStockFLRank.hpp"
#include "DataSetStockFLRankCV.hpp"
#include "RankSet.hpp"
#include <fstream>

void test_stock() {
  std::cout << "test_stock" << std::endl;
  RankSet train = DataSetStockFLRank::get_train();
  RankSet test = DataSetStockFLRank::get_test();

  std::shared_ptr<NeuralNetworkRankNet> network(new NeuralNetworkRankNet(DataSetStockFLRank::get_feature_size(), 10));

  for (int i = 0; i < 50; i++)
    network->train(train);
  network->rank(train);
  network->rank(test);

  //std::cout << network->test_pair(train) << std::endl;
  //std::cout << network->test_pair(test) << std::endl;

  std::unordered_map<std::size_t, RankList> map = test.get_map();

  double rough = 0;
  double count = 0;
  for (std::unordered_map<std::size_t, RankList>::iterator it = map.begin(); it != map.end(); it++) {
    rough += it->second.get_normalized_discounted_cumulative_gain();
    count++;
  }

  std::cout << "rough average ncdg: " << rough / count << std::endl;

  std::ofstream out("/data/wchan/research/flrank/data/prediction.out");

  // probably one of the ugliest code ive written in my life...
  // but as a phd student.... blah excuses!
  for (std::size_t i = 0; i < DataSetStockFLRank::m_test_unsorted.size(); i++) {
    double r = network->f0(DataSetStockFLRank::m_test_unsorted[i])[0];
    out << r << std::endl;
  }

  out.close();
}

void test_stock_cv(int mod) {
  for (std::size_t i = 0; i < 10; i++) {
    RankSet train = DataSetStockFLRankCV::get_train(i, mod);

    std::shared_ptr<NeuralNetworkRankNet> network(new NeuralNetworkRankNet(DataSetStockFLRank::get_feature_size(), 10));

    for (int k = 0; k < 50; k++)
      network->train(train);

    std::stringstream xxx;
    xxx << "/data/wchan/research/flrank/ranknet-stock-labels-cv-prediction-" << mod << "-" << i;

    std::string filename = xxx.str();
    std::ofstream out(filename);

    std::vector<boost::numeric::ublas::vector<double> > test = DataSetStockFLRankCV::get_test(i, mod);
    for (std::size_t j = 0; j < test.size(); j++) {
      double r = network->f0(test[j])[0];
      out << r << std::endl;
    }

    out.close();
  }
}

int main(int argc, char* argv[]) {
  std::cout.setf(std::ios_base::fixed);
  std::cout.precision(6);

  int mod = atoi(argv[2]);


  // vivian();
  // gladys();
  // vivian();
  test_stock_cv(mod);

  //test_RankNet();
  //test_LambdaRank();

  //init_constants();
  //omp();

  //test_backpropagation();
  //test_mnist();
  //test_rprop_xor();
  //test_rprop_mnist();
  //test_cuda();

  return 1;
}
