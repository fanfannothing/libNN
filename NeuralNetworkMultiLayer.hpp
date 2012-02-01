/*
 * NeuralNetworkMultiLayer.hpp
 *
 *  Created on: Jan 21, 2012
 *      Author: wchan
 */

#ifndef NEURALNETWORKMULTILAYER_HPP_
#define NEURALNETWORKMULTILAYER_HPP_

#include "NeuralNetworkLayer.hpp"
#include "NeuralNetworkLayerConstant.hpp"

template<class ActivationFunction>
class NeuralNetworkMultiLayer : public NeuralNetwork {
public:
  /* the first entry is for the input layer... therefore a "two-layer" network actually has 3 entries */
  NeuralNetworkMultiLayer(std::vector<std::size_t> size) {
    assert(size.size() >= 2);

    m_input.reset(new NeuralNetworkLayerConstant(size[0]));

    m_layers.resize(size.size() - 1);

    m_layers[0].reset(new NeuralNetworkLayer<ActivationFunction>(size[1], m_input));
    // TODO: switch this to iterators?
    for (std::size_t i = 2; i < size.size(); i++) {
      m_layers[i - 1].reset(new NeuralNetworkLayer<ActivationFunction>(size[i], m_layers[i - 2]));
    }
  }

  virtual ~NeuralNetworkMultiLayer() {
  }

  virtual void set_value(boost::numeric::ublas::vector<double> value) {
    m_input->set_value(value);
  }

  virtual void compute() {
    for (std::size_t i = 0; i < m_layers.size(); i++) {
      m_layers[i]->compute();
    }
  }

  virtual boost::numeric::ublas::vector<double> get_outputs() {
    return m_layers[m_layers.size() - 1]->get_outputs();
  }

  virtual boost::numeric::ublas::vector<double> f(boost::numeric::ublas::vector<double> in) {
    set_value(in);
    compute();
    return this->get_outputs();
  }

  virtual std::vector<std::shared_ptr<NeuralNetworkLayer<ActivationFunction> > > get_layers() {
    return m_layers;
  }

  virtual std::shared_ptr<NeuralNetworkLayerConstant> get_layer_input() {
    return m_input;
  }

  virtual void print() {
    // std::cerr << "MSE: " << mse() << std::endl;
    std::cerr << "Input Layer" << std::endl;
    m_input->print();

    for (std::size_t i = 0; i < (m_layers.size() - 1); i++) {
      std::cerr << "Hidden Layer" << std::endl;
      m_layers[i]->print();
    }

    std::cerr << "Output Layer" << std::endl;
    m_layers[m_layers.size() - 1]->print();

    std::cerr << std::endl;
  }

  virtual NeuralNetworkMultiLayer<ActivationFunction>* clone() {
    NeuralNetworkMultiLayer<ActivationFunction>* clone = new NeuralNetworkMultiLayer<ActivationFunction>();

    clone->m_mse = m_mse;
    clone->m_input.reset(m_input->clone());

    std::shared_ptr<NeuralNetwork> prev = clone->m_input;
    for (std::size_t i = 0; i < m_layers.size(); i++) {
      std::shared_ptr<NeuralNetworkLayer<ActivationFunction> > layer(m_layers[i]->clone());
      layer->set_inputs(prev);

      clone->m_layers.push_back(layer);

      prev = layer;
    }

    return clone;
  }

protected:
  NeuralNetworkMultiLayer() {
  }

  std::shared_ptr<NeuralNetworkLayerConstant> m_input;
  std::vector<std::shared_ptr<NeuralNetworkLayer<ActivationFunction> > > m_layers;
};

#endif /* NEURALNETWORKMULTILAYER_HPP_ */
