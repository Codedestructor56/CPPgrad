#ifndef MLP_H
#define MLP_H

#include "autograd.h"
#include <vector>
#include <string>

struct Neuron {
    public:
      std::vector<Data> weights;
      Data bias;
      
      Data result;
      Neuron(int num_inputs);

      Data forward(const std::vector<Data>& inputs);

      Data backward();
};

class Layer {
public:
    std::vector<Neuron> neurons;

    Layer(int num_neurons, int num_inputs_per_neuron);

    std::vector<Data> forward(const std::vector<Data>& inputs);

    void backward();
};

class MLP {
public:
    std::vector<Layer> layers;

    MLP(const std::vector<int>& layer_sizes);

    std::vector<Data> forward(const std::vector<Data>& inputs);
    void backward();

    std::string summary();
};
#endif
