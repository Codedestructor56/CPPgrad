#include "mlp.h"
#include <cstdlib>
#include <ctime>
#include <cmath>

// Helper function to initialize weights with random values
Data randomWeight() {
    return Data(static_cast<double>(rand()) / RAND_MAX * 2.0 - 1.0);
}

Neuron::Neuron(int num_inputs):bias(0.0f), result(0.0f) {
    for (int i = 0; i < num_inputs; ++i) {
        weights.push_back(randomWeight());
    }
    bias = randomWeight();
}

Data Neuron::forward(const std::vector<Data>& inputs) {
    Data sum(bias.getGrad());
    for (size_t i = 0; i < inputs.size(); ++i) {
        sum += (inputs[i] * weights[i]);
    }
    result = sum;
    return result;
}

Data Neuron::backward() {
    Data res = result.sigmoid();
    res.backward();
    for (Data& weight : weights) {
        weight.setGrad(weight.getGrad() + res.getGrad());
    }
    bias.setGrad(bias.getGrad() + res.getGrad());
    return res;
}

Layer::Layer(int num_neurons, int num_inputs_per_neuron) {
    for (int i = 0; i < num_neurons; ++i) {
        neurons.emplace_back(num_inputs_per_neuron);
    }
}

std::vector<Data> Layer::forward(const std::vector<Data>& inputs) {
    std::vector<Data> outputs;
    for (Neuron& neuron : neurons) {
        outputs.push_back(neuron.forward(inputs));
    }
    return outputs;
}

void Layer::backward() {
    for (Neuron& neuron : neurons) {
        neuron.backward();
    }
}

MLP::MLP(const std::vector<int>& layer_sizes) {
    for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
        layers.emplace_back(layer_sizes[i + 1], layer_sizes[i]);
    }
}

std::vector<Data> MLP::forward(const std::vector<Data>& inputs) {
    std::vector<Data> current_inputs = inputs;
    for (Layer& layer : layers) {
        current_inputs = layer.forward(current_inputs);
    }
    return current_inputs;
}

void MLP::backward() {
    for (Layer& layer : layers) {
        layer.backward();
    }
}
