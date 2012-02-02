/*
 * NeuralNetworkSparse.hpp
 *
 *  Created on: Feb 1, 2012
 *      Author: wchan
 */

#ifndef NEURALNETWORKSPARSE_HPP_
#define NEURALNETWORKSPARSE_HPP_

#include "NeuralNetwork.hpp"
#include <memory>
#include <random>
#include <cmath>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <algorithm>
#include <functional>

/*
 * sparsely connected feed forward neural network mesh
 *
 * underlying implementation may or may not be a sparse matrix
 */
template<class ActivationFunction>
class NeuralNetworkSparse : public NeuralNetwork {
public:
  NeuralNetworkSparse(std::size_t count, std::shared_ptr<NeuralNetwork> in) :
      NeuralNetwork(in) {
    m_weights.resize(count, count + in->get_outputs_size(), false);
    m_outputs.resize(count + in->get_outputs_size(), false);
    m_dydx.resize(count + in->get_outputs_size(), false);
    m_dedx.resize(count + in->get_outputs_size(), false);

    m_weights.clear();
    m_outputs.clear();
    m_dedx.clear();
    m_dydx.clear();

    // generate random numbers uniformly between -1 and 1; however the weight matrix is still not valid because it allows recurrent connections...f
    std::generate(m_weights.data().begin(), m_weights.data().end(), std::bind(std::uniform_real_distribution<double>(-1, 1), mt));
  }

  ~NeuralNetworkSparse() {
  }

  virtual void compute() {
    // need to concat current output with previous output
    boost::numeric::ublas::vector<double> inputs(m_weights.size2());
    std::copy(m_outputs.begin(), m_outputs.end(), inputs.begin());
    std::copy(m_prev->get_outputs().begin(), m_prev->get_outputs().end(), inputs.begin() + m_outputs.size());

    // perform weight calculation
    boost::numeric::ublas::vector<double> activations = prod(m_weights, inputs);

    // element wise activation
    std::transform(activations.begin(), activations.end(), m_outputs.begin(), std::ptr_fun(&ActivationFunction::f));

    // calculate derivative while we are at it...
    std::transform(activations.begin(), activations.end(), m_outputs.begin(), m_dydx.begin(), std::ptr_fun(&ActivationFunction::d));

    // guarantee the convergence?
    if (!std::equal(m_outputs.begin(), m_outputs.end(), inputs.begin())) compute();
  }

  /**
   * pass in a weight mask -- weight mask should be the same dimension as weights
   *
   * a 1.0 means connection; a 0.0 means no connection
   */
  virtual void set_weight_mask(boost::numeric::ublas::matrix<double> mask) {
    m_weights_mask = mask;
    mask();
  }

  /**
   * after every weight update.. this function should be called to zero out any weights that have been accidently changed that should be zeroed out
   */
  virtual void mask() {
    std::transform(m_weights.data().begin(), m_weights.data().end(), m_weights_mask.data().begin(), m_weights.data().begin(), std::multiplies<double>());
  }

  /**
   * rows is neurons in the network
   * cols is inputs + neurons in the network
   */
  virtual boost::numeric::ublas::matrix<double>& weights() {
    return m_weights;
  }

protected:
  boost::numeric::ublas::matrix<double> m_weights_mask;
  boost::numeric::ublas::matrix<double> m_weights;
};

#endif /* NEURALNETWORKSPARSE_HPP_ */
