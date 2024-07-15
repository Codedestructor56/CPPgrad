#include <iostream>
#include "autograd.h"
#include <unordered_set>
#include "mlp.h"

void printVector(const std::vector<Data>& vec) {
    for (const auto& data : vec) {
        std::cout << data.getData() << " ";
    }
    std::cout << std::endl;
}

int main() {
    // Create a simple MLP with 3 layers
    std::vector<int> layer_sizes = {3, 4, 2};  // 3 inputs, one hidden layer with 4 neurons, and 2 outputs
    MLP mlp(layer_sizes);

    // Create input data
    std::vector<Data> inputs = { Data(1.0), Data(2.0), Data(3.0) };

    // Forward pass through the network
    std::vector<Data> outputs = mlp.forward(inputs);

    // Print input values
    std::cout << "Input values: ";
    printVector(inputs);

    // Print outputs of each layer
    std::cout << "Outputs at each layer:" << std::endl;
    std::vector<Data> layer_input = inputs;
    for (size_t i = 0; i < mlp.layers.size(); ++i) {
        std::vector<Data> layer_output = mlp.layers[i].forward(layer_input);
        std::cout << "Layer " << i + 1 << ": ";
        printVector(layer_output);
        layer_input = layer_output;
    }

    // Print final output
    std::cout << "Final output: ";
    printVector(outputs);

    return 0;
}
