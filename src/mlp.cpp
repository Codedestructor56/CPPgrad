#include "mlp.h"

Neuron::Neuron(int num_inputs) : bias(0.0) {
    for (int i = 0; i < num_inputs; ++i) {
        weights.emplace_back(0.0);  
    }
}

Data Neuron::forward(const std::vector<Data>& inputs) {
    Data total = bias;
    for (size_t i = 0; i < inputs.size(); ++i) {
        total = total + (inputs[i] * weights[i]);
    }
    return total.sigmoid();  
}

Layer::Layer(int num_neurons, int num_inputs_per_neuron) {
    for (int i = 0; i < num_neurons; ++i) {
        neurons.emplace_back(num_inputs_per_neuron);
    }
}

std::vector<Data> Layer::forward(const std::vector<Data>& inputs) {
    std::vector<Data> outputs;
    for (auto& neuron : neurons) {
        outputs.push_back(neuron.forward(inputs));
    }
    return outputs;
}

MLP::MLP(const std::vector<int>& layer_sizes) {
    for (size_t i = 1; i < layer_sizes.size(); ++i) {
        layers.emplace_back(layer_sizes[i], layer_sizes[i - 1]);
    }
}

std::vector<Data> MLP::forward(const std::vector<Data>& inputs) {
    std::vector<Data> output = inputs;
    for (auto& layer : layers) {
        output = layer.forward(output);
    }
    return output;
}
