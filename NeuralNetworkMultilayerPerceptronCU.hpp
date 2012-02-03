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

template<class ActivationFunction = ActivationFunctionTanh>
class NeuralNetworkMultilayerPerceptronCU : public NeuralNetworkCU {
public:
  /* the first entry is for the input layer... therefore a "two-layer" network actually has 3 entries */
  NeuralNetworkMultilayerPerceptronCU(std::vector<size_t> size) {
    assert(size.size() >= 2);

    m_input.reset(new NeuralNetworkLayerConstantCU(size[0]));

    m_layers.resize(size.size() - 1);

    m_layers[0].reset(new NeuralNetworkLayerCU<ActivationFunction>(size[1], m_input));
    // TODO: switch this to iterators?
    for (size_t i = 2; i < size.size(); i++) {
      m_layers[i - 1].reset(new NeuralNetworkLayerCU<ActivationFunction>(size[i], m_layers[i - 2]));
    }
  }

  NeuralNetworkMultilayerPerceptronCU(std::shared_ptr<NeuralNetworkMultilayerPerceptron<ActivationFunction> > network) {
    m_input.reset(new NeuralNetworkLayerConstantCU(network->get_layer_input()));

    m_layers.resize(network->get_layers().size());
    m_layers[0].reset(new NeuralNetworkLayerCU<ActivationFunction>(network->get_layers()[0], m_input));

    for (size_t i = 1; i < m_layers.size(); i++) {
      m_layers[i].reset(new NeuralNetworkLayerCU<ActivationFunction>(network->get_layers()[i], m_layers[i - 1]));
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

  virtual void get_outputs(boost::numeric::ublas::vector<double>& outputs) {
    m_layers[m_layers.size() - 1]->get_outputs(outputs);
  }

  virtual double* f(boost::numeric::ublas::vector<double> in, boost::numeric::ublas::vector<double>& outputs) {
    double* cuda_ptr = f(in);
    get_outputs(outputs);
    return cuda_ptr;
  }

  virtual std::shared_ptr<NeuralNetworkLayerConstantCU> get_layer_input() {
    return m_input;
  }

  virtual std::vector<std::shared_ptr<NeuralNetworkLayerCU<ActivationFunction> > > get_layers() {
    return m_layers;
  }

protected:
  NeuralNetworkMultilayerPerceptronCU() {
  }

  std::shared_ptr<NeuralNetworkLayerConstantCU> m_input;
  std::vector<std::shared_ptr<NeuralNetworkLayerCU<ActivationFunction> > > m_layers;
};

#endif /* NEURALNETWORKMULTILAYERPERCEPTRON_HPP_ */
