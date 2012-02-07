/*
 * NeuralNetworkMultiLayer.hpp
 *
 *  Created on: Jan 21, 2012
 *      Author: wchan
 */

#ifndef NEURALNETWORKMULTILAYERPERCEPTRON_HPP_
#define NEURALNETWORKMULTILAYERPERCEPTRON_HPP_

#include "NeuralNetworkLayer.hpp"
#include "NeuralNetworkLayerConstant.hpp"
#include "ActivationFunctionTanh.hpp"

class NeuralNetworkMultilayerPerceptron : public NeuralNetwork {
public:
  /* the first entry is for the input layer... therefore a "two-layer" network actually has 3 entries */
  NeuralNetworkMultilayerPerceptron(std::vector<std::size_t> size) {
    assert(size.size() >= 1);

    m_input.reset(new NeuralNetworkLayerConstant(size[0]));

    m_layers.resize(size.size() - 1);

    /* we default to use tanh activation functions */
    if (m_layers.size() <= 0) return;

    m_layers[0].reset(new NeuralNetworkLayer(size[1], m_input, std::shared_ptr<ActivationFunction>(new ActivationFunctionTanh())));
    // TODO: switch this to iterators?
    for (std::size_t i = 2; i < size.size(); i++) {
      m_layers[i - 1].reset(new NeuralNetworkLayer(size[i], m_layers[i - 2], std::shared_ptr<ActivationFunction>(new ActivationFunctionTanh())));
    }
  }

  virtual ~NeuralNetworkMultilayerPerceptron() {
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

  virtual void add_layer(std::shared_ptr<NeuralNetworkLayer> layer) {
    if (m_layers.size() == 0)
      layer->set_inputs(m_input);
    else
      layer->set_inputs(m_layers[m_layers.size() - 1]);

    m_layers.push_back(layer);
  }

  virtual std::vector<std::shared_ptr<NeuralNetworkLayer> > get_layers() {
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

  /**
   * creates a new NeuralNetworkMultilayerPerceptron with the same structure, weights, and activation function
   *
   * output values including derivatives and errors are not cloned however
   */
  virtual NeuralNetworkMultilayerPerceptron* clone() {
    NeuralNetworkMultilayerPerceptron* clone = new NeuralNetworkMultilayerPerceptron();

    // our activation is empty actually...
    // clone->m_activation.reset(m_activation->clone());
    clone->m_input.reset(m_input->clone());

    std::cerr << std::flush;
    std::shared_ptr<NeuralNetwork> prev(clone->m_input);
    for (std::size_t i = 0; i < m_layers.size(); i++) {
      std::shared_ptr<NeuralNetworkLayer> layer(m_layers[i]->clone());
      layer->set_inputs(prev);

      clone->m_layers.push_back(layer);

      prev = layer;
    }

    return clone;
  }

protected:
  NeuralNetworkMultilayerPerceptron() {
  }

  std::shared_ptr<NeuralNetworkLayerConstant> m_input;
  std::vector<std::shared_ptr<NeuralNetworkLayer> > m_layers;
};

#endif /* NEURALNETWORKMULTILAYERPERCEPTRON_HPP_ */
