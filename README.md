# CPPgrad

This project is inspired by Andrej Karpathy's **micrograd**, providing a simple implementation of a multi-layer perceptron (MLP) with automatic differentiation. It showcases fundamental concepts of neural networks and the backpropagation algorithm while leveraging C++ and Pybind11 for seamless integration with Python.

## Project Structure

- **`bindings.cpp`**: The Pybind11 bindings that expose the C++ classes and functions to Python.
- **`autograd.h` / `autograd.cpp`**: Implementations of the `Data` class for automatic differentiation, including various operations and activation functions.
- **`mlp.h` / `mlp.cpp`**: Definitions of the `Neuron`, `Layer`, and `MLP` classes, which together form the structure of the neural network.
- **`mlp.py`**: A test script to demonstrate usage of the MLP module from Python.

## Requirements

- C++11 or later
- CMake
- Python 3.x
- Pybind11

## Building the Project

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd CPPgrad



2. Create a build directory, and just run the following commands:

<pre><code>
mkdir build
cd build
cmake ..
make
</code></pre>

<h2>Features</h2>

<ul>
  <li><strong>Automatic Differentiation</strong>: Supports basic operations (+, *, -, /) and computes gradients using backpropagation.</li>
  <li><strong>Custom Activation Functions</strong>: Implementations of sigmoid, ReLU, tanh, swish, and gelu.</li>
  <li><strong>Flexible Architecture</strong>: Easily create multi-layer perceptrons by specifying the number of neurons in each layer.</li>
</ul>

<h2>Future Improvements</h2>

<ul>
  <li>Implement additional optimization algorithms.</li>
  <li>Expand the functionality to include convolutional layers.</li>
  <li>Create a more robust testing framework.</li>
</ul>



