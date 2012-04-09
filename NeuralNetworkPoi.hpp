/*
 * NeuralNetworkPoi.hpp
 *
 *  Created on: Feb 22, 2012
 *      Author: wchan
 */

#ifndef NEURALNETWORKPOI_HPP_
#define NEURALNETWORKPOI_HPP_

#include "NeuralNetworkMultilayerPerceptron.hpp"
#include "NeuralNetworkFunctionApproximator.hpp"
#include "Backpropagation.hpp"
#include "DataSetMNIST.hpp"
#include <boost/numeric/ublas/vector.hpp>
#include <png++/png.hpp>
#include <sstream>

class NeuralNetworkPoi : public NeuralNetwork {
public:
  NeuralNetworkPoi() {
    index = 0;
    // m_network.reset(new NeuralNetworkFunctionApproximator(DataSetMNIST::get_feature_size(), DataSetMNIST::get_feature_size() / 2, DataSetMNIST::get_feature_size()));
    m_network.reset(new NeuralNetworkMultilayerPerceptron( { DataSetMNIST::get_feature_size(), DataSetMNIST::get_feature_size() / 8, DataSetMNIST::get_feature_size() }));
    m_network_medium.reset(new NeuralNetworkMultilayerPerceptron( { DataSetMNIST::get_feature_size(), DataSetMNIST::get_feature_size() / 8, DataSetMNIST::get_feature_size() }));
    m_network_deep.reset(new NeuralNetworkMultilayerPerceptron( { DataSetMNIST::get_feature_size(), 300, DataSetMNIST::get_label_size() }));
  }

  void train() {
    train_shallow();
    train_deep();
  }

  void train_deep() {
    std::cerr << "train_deep" << std::endl;
    std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > train = DataSetMNIST::get_train();
    std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > train2;

    train.resize(1000);

    for (std::size_t i = 0; i < train.size(); i++) {
      train2.push_back(std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> >(f_shallow(train[i].first), train[i].second));
    }

    Backpropagation<>::train(m_network_deep, train2, 10, 0.001, 0.001);
    std::cerr << "deep 0 done" << std::endl;
    Backpropagation<>::train(m_network_deep, train2, 10, 0.001, 0.0001);
    std::cerr << "deep 1 done" << std::endl;
    // Backpropagation<>::train(m_network_deep, train2, 10, 0.001, 0.00001);
  }

  void train_shallow() {
    std::cout << "train_autoencoder" << std::endl;
    std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > train = DataSetMNIST::get_train();
    std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > train2;
    std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > train3;

    train.resize(1000);

    std::vector<boost::numeric::ublas::vector<double> > labels(10);
    std::vector<double> count(10);
    for (std::size_t i = 0; i < labels.size(); i++) {
      labels[i].resize(DataSetMNIST::get_feature_size());
    }

    for (std::size_t i = 0; i < train.size(); i++) {
      for (std::size_t j = 0; j < labels.size(); j++) {
        if (train[i].second[j] == 1.0) {
          labels[j] += train[i].first;
          count[j] += 1;
        }
      }
    }

    for (std::size_t i = 0; i < labels.size(); i++) {
      labels[i] /= count[i];
      for (std::size_t j = 0; j < labels[i].size(); j++) {
        if (labels[i][j] > 1.0) {
          std::cerr << labels[i][j] << std::endl;
          throw;
        }
        if (labels[i][j] < -1.0) {
          std::cerr << labels[i][j] << std::endl;
          throw;
        }
      }
    }

    png::image<png::rgb_pixel> image(28 * 10, 28);
    for (int z = 0; z < labels.size(); z++) {
      for (std::size_t y = 0; y < 28; ++y) {
        for (std::size_t x = 0; x < 28; ++x) {
          double c = labels[z][y * 28 + x] + 1;
          c *= 127.5;
          char b = c;
          image[y][x + z * 28] = png::rgb_pixel(b, b, b);
        }
      }
    }
    image.write("cool.png");

    for (std::size_t i = 0; i < train.size(); i++) {
      // probably cheaper to just multiple all instead of branching
      boost::numeric::ublas::vector<double> label(DataSetMNIST::get_feature_size());

      for (std::size_t j = 0; j < labels.size(); j++) {
        if (train[i].second[j] == 1.0) {
          label = labels[j];
        }
      }

      train2.push_back(std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> >(train[i].first, label));
    }

    for (int i = 0; i < 10; i++) {
      train3.push_back(std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> >(labels[i], labels[i]));
    }

    std::cerr << "labels created" << std::endl;

    //Backpropagation<>::train(m_network, train2, 10, 0.001, 0.1);
    // Backpropagation<>::train(m_network, train2, 10, 0.001, 0.01);
    BackpropagationCE<>::train(m_network, train2, 10, 0.001, 0.001);
    BackpropagationCE<>::train(m_network, train2, 10, 0.001, 0.0001);
    // BackpropagationCE<>::train(m_network, train2, 10, 0.001, 0.00001);
    // Backpropagation<>::train(m_network, train2, 10, 0.001, 0.00001);

    BackpropagationCE<>::train(m_network_medium, train3, 10, 0.001, 0.001);
    BackpropagationCE<>::train(m_network_medium, train3, 10, 0.001, 0.0001);
    BackpropagationCE<>::train(m_network_medium, train3, 10, 0.001, 0.00001);

    std::cerr << "done train autoencoder" << std::endl;
  }

  virtual boost::numeric::ublas::vector<double> f(boost::numeric::ublas::vector<double> x) {
    boost::numeric::ublas::vector<double> shallow = f_shallow(x);

    return m_network_deep->f(shallow);
  }

  int index;

  virtual boost::numeric::ublas::vector<double> f_shallow(boost::numeric::ublas::vector<double> x) {
    double epsilon = 1e-6;

    std::stringstream filename;
    filename << "cool_morph_" << index++ << ".png";

    int wee = 0;
    png::image<png::rgb_pixel> image(28 * 50, 28);
    boost::numeric::ublas::vector<double> t = m_network->f(x);

    for (std::size_t y = 0; y < 28; ++y) {
      for (std::size_t z = 0; z < 28; ++z) {
        double c = x[y * 28 + z] + 1;
        c *= 127.5;
        char b = c;
        image[y][wee * 28 + z] = png::rgb_pixel(b, b, b);
      }
    }
    wee++;

    for (std::size_t y = 0; y < 28; ++y) {
      for (std::size_t z = 0; z < 28; ++z) {
        double c = t[y * 28 + z] + 1;
        c *= 127.5;
        char b = c;
        image[y][wee * 28 + z] = png::rgb_pixel(b, b, b);
      }
    }
    wee++;

    double norm = norm_2(t - x);

    int i = 0;
    while (norm > epsilon) {
      x = t;
      t = m_network->f(t);
      for (std::size_t y = 0; y < 28; ++y) {
        for (std::size_t z = 0; z < 28; ++z) {
          double c = t[y * 28 + z] + 1;
          c *= 127.5;
          char b = c;
          image[y][wee * 28 + z] = png::rgb_pixel(b, b, b);
        }
      }
      wee++;

      norm = norm_2(t - x);
      if (i++ > 50) {
        std::cout << "fail" << std::endl;
        std::cerr << "fail" << std::endl;
        break;
      };
    }

    if (index < 100) image.write(filename.str().c_str());

    // std::cout << "pass" << std::endl;
    return t;
  }

  virtual void test2() {
    std::vector<std::pair<boost::numeric::ublas::vector<double>, boost::numeric::ublas::vector<double> > > train = DataSetMNIST::get_train();

    train.resize(100);

    png::image<png::rgb_pixel> image(28 * train.size(), 28 * 2);
    for (int z = 0; z < train.size(); z++) {
      boost::numeric::ublas::vector<double> f = f_shallow(train[z].first);

      for (std::size_t y = 0; y < 28; ++y) {
        for (std::size_t x = 0; x < 28; ++x) {
          double c = f[y * 28 + x] + 1;
          c *= 127.5;
          char b = c;
          image[y + 28][x + z * 28] = png::rgb_pixel(b, b, b);

          c = train[z].first[y * 28 + x] + 1;
          c *= 127.5;
          b = c;
          image[y][x + z * 28] = png::rgb_pixel(b, b, b);
        }
      }
    }
    image.write("coolbadass.png");
  }

  virtual void compute() {
    throw;
  }

  virtual std::size_t get_outputs_size() {
    return DataSetMNIST::get_label_size();
  }

protected:
  std::shared_ptr<NeuralNetworkMultilayerPerceptron> m_network;
  std::shared_ptr<NeuralNetworkMultilayerPerceptron> m_network_medium;
  std::shared_ptr<NeuralNetworkMultilayerPerceptron> m_network_deep;
}
;

#endif /* NEURALNETWORKPOI_HPP_ */
