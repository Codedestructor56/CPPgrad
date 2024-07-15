#include <iostream>
#include "autograd.h"
#include "mlp.h"
#include <unordered_set>
#include <vector>

void printVector(const std::vector<Data>& vec) {
    for (const auto& data : vec) {
        std::cout << "Data: " << data.getData() << " \n";
        std::cout << "Grad: " << data.getGrad() << " \n";
    }
    std::cout << std::endl;
}

void traverseGraph(Data* node, std::unordered_set<Data*>& visited) {
    if (!node || visited.find(node) != visited.end()) {
        return;
    }

    visited.insert(node);
    std::cout << "Node data: " << node->getData() << ", Grad: " << node->getGrad() << std::endl;

    for (Data* child : node->getChildren()) {
        std::cout << "    " << node->getData() << " -> " << child->getData() << std::endl;
        traverseGraph(child, visited);
    }
}

int main() {
    // Demonstrate a Neuron
    Neuron neuron(3);
    std::vector<Data> inputs = {Data(1.0), Data(2.0), Data(3.0)};
    Data neuron_output = neuron.forward(inputs);
    std::cout << "Neuron Output: " << neuron_output.getData() << std::endl;
    neuron_output.backward();
    std::cout << "Neuron backward pass:" << std::endl;
    std::unordered_set<Data*> visited1;
    traverseGraph(&neuron_output, visited1);

    // Demonstrate a Layer
    Layer layer(2, 3);
    std::vector<Data> layer_output = layer.forward(inputs);
    std::cout << "Layer Output:" << std::endl;
    printVector(layer_output);
    layer.backward();
    std::cout << "Layer backward pass:" << std::endl;
    for (auto& output : layer_output) {
        std::unordered_set<Data*> visited2;
        traverseGraph(&output, visited2);
    }

    // Demonstrate an MLP
    std::vector<int> layer_sizes = {3, 2, 1}; // 3 inputs, 1 hidden layer with 2 neurons, 1 output neuron
    MLP mlp(layer_sizes);
    std::vector<Data> mlp_output = mlp.forward(inputs);
    std::cout << "MLP Output:" << std::endl;
    std::cout << "mlp forward pass:" << std::endl;

    for (auto& output : mlp_output) {
        std::unordered_set<Data*> visited3;
        traverseGraph(&output, visited3);
    }
    
    //mlp.layers[0].backward();    
    printVector(mlp_output);
    mlp.backward();
    std::cout << "mlp backward pass:" << std::endl;
    for (auto& output : mlp_output) {
        std::unordered_set<Data*> visited4;
        traverseGraph(&output, visited4);
    }

    return 0;
}
