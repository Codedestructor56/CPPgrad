#ifndef MLP_H
#define MLP_H

#include "autograd.h"
#include <vector>

struct Neuron {
    std::vector<Data> weights;
    Data bias;

    Neuron(int num_inputs);

    Data forward(const std::vector<Data>& inputs);
};

class Layer {
public:
    std::vector<Neuron> neurons;

    Layer(int num_neurons, int num_inputs_per_neuron);

    std::vector<Data> forward(const std::vector<Data>& inputs);
};

class MLP {
public:
    std::vector<Layer> layers;

    MLP(const std::vector<int>& layer_sizes);

    std::vector<Data> forward(const std::vector<Data>& inputs);
};
#endif

