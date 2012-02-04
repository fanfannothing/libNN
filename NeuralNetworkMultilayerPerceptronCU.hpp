/*
 * NeuralNetworkMultiLayerCU.hpp
 *
 *  Created on: Jan 31, 2012
 *      Author: wchan
 */

#ifndef NEURALNETWORKMULTILAYERPERCEPTRONCU_HPP_
#define NEURALNETWORKMULTILAYERPERCEPTRONCU_HPP_

#include "NeuralNetworkMultilayerPerceptron.hpp"
#include "NeuralNetworkLayerCU.hpp"
#include "NeuralNetworkLayerConstantCU.hpp"
#include "ActivationFunctionTanh.hpp"

class NeuralNetworkMultilayerPerceptronCU : public NeuralNetworkCU {
public:
  /* the first entry is for the input layer... therefore a "two-layer" network actually has 3 entries */
  NeuralNetworkMultilayerPerceptronCU(std::vector<size_t> size) {
    assert(size.size() >= 2);

    m_input.reset(new NeuralNetworkLayerConstantCU(size[0]));

    m_layers.resize(size.size() - 1);

    /* we default to use tanh activation functions */
    if (m_layers.size() <= 0) return;

    m_layers[0].reset(new NeuralNetworkLayerCU(size[1], m_input, std::shared_ptr<ActivationFunction>(new ActivationFunctionTanh())));
    // TODO: switch this to iterators?
    for (size_t i = 2; i < size.size(); i++) {
      m_layers[i - 1].reset(new NeuralNetworkLayerCU(size[i], m_layers[i - 2], std::shared_ptr<ActivationFunction>(new ActivationFunctionTanh())));
    }
  }

  NeuralNetworkMultilayerPerceptronCU(std::shared_ptr<NeuralNetworkMultilayerPerceptron> network) {
    m_input.reset(new NeuralNetworkLayerConstantCU(network->get_layer_input()));

    m_layers.resize(network->get_layers().size());
    m_layers[0].reset(new NeuralNetworkLayerCU(network->get_layers()[0], m_input));

    for (size_t i = 1; i < m_layers.size(); i++) {
      m_layers[i].reset(new NeuralNetworkLayerCU(network->get_layers()[i], m_layers[i - 1]));
    }
  }

  virtual ~NeuralNetworkMultilayerPerceptronCU() {
  }

  virtual void set_value(boost::numeric::ublas::vector<double> value) {
    m_input->set_value(value);
  }

  virtual void compute() {
    for (size_t i = 0; i < m_layers.size(); i++) {
      m_layers[i]->compute();
    }
  }

  virtual double* get_outputs() {
    return m_layers[m_layers.size() - 1]->get_outputs();
  }

  virtual double* f(boost::numeric::ublas::vector<double> in) {
    set_value(in);
    compute();

    return get_outputs();
  }

  virtual double* get_outputs(boost::numeric::ublas::vector<double>& outputs) {
    m_layers[m_layers.size() - 1]->get_outputs(outputs);
    return get_outputs();
  }

  virtual double* f(boost::numeric::ublas::vector<double> in, boost::numeric::ublas::vector<double>& outputs) {
    double* cuda_ptr = f(in);
    get_outputs(outputs);
    return cuda_ptr;
  }

  virtual void add_layer(std::shared_ptr<NeuralNetworkLayerCU> layer) {
    if (m_layers.size() == 0)
      layer->set_inputs(m_input);
    else
      layer->set_inputs(m_layers[m_layers.size() - 1]);

    m_layers.push_back(layer);
  }

  virtual std::shared_ptr<NeuralNetworkLayerConstantCU> get_layer_input() {
    return m_input;
  }

  virtual std::vector<std::shared_ptr<NeuralNetworkLayerCU> > get_layers() {
    return m_layers;
  }

protected:
  NeuralNetworkMultilayerPerceptronCU() {
  }

  std::shared_ptr<NeuralNetworkLayerConstantCU> m_input;
  std::vector<std::shared_ptr<NeuralNetworkLayerCU> > m_layers;
};

#endif /* NEURALNETWORKMULTILAYERPERCEPTRON_HPP_ */
