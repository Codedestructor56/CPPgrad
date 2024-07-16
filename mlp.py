import CPPgrad
import numpy as np

# Define a simple dataset (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Convert the dataset to CPPgrad.Data objects
def to_data(arr):
    return [CPPgrad.Data(float(a)) for a in arr]

X_data = [to_data(x) for x in X]
y_data = [CPPgrad.Data(float(label)) for label in y]

# Initialize the MLP
layers = [2, 4, 1]  # Example: 2 input neurons, 4 hidden neurons, 1 output neuron
mlp = CPPgrad.MLP(layers)

# Forward pass
def forward_pass(mlp, inputs):
    outputs = mlp.forward(inputs)
    return outputs


def compute_loss(predicted, actual):
    loss = CPPgrad.Data(0.0)  # Initialize loss as a Data object
    for p, a in zip(predicted, actual):
        if isinstance(p, CPPgrad.Data) and isinstance(a, CPPgrad.Data):
            loss += (p - a) * (p - a)
        else:
            raise TypeError("Both predicted and actual must be CPPgrad.Data objects")
    return loss
# Compute the loss (Mean Squared Error)
def compute_loss(predicted, actual):
    loss = CPPgrad.Data(0.0)
    for p, a in zip(predicted, actual):
        loss += (p - a) * (p - a)
    return loss

# Training loop
epochs = 1000
learning_rate = 0.01

for epoch in range(epochs):
    total_loss = CPPgrad.Data(0.0)
    for inputs, target in zip(X_data, y_data):
        # Forward pass
        outputs = forward_pass(mlp, inputs)
        
        # Compute loss
        loss = compute_loss(outputs, [target])
        total_loss += loss
        
        # Backward pass
        loss.backward()
        
        # Update weights (simple gradient descent)
        for layer in mlp.getLayers():
            for neuron in layer.getNeurons():
                for weight in neuron.getWeights():
                    weight.setData(weight.getData() - learning_rate * weight.getGrad())
                    weight.setGrad(0.0)
    
    # Reset gradients after each epoch
    for inputs in X_data:
        for data in inputs:
            data.setGrad(0.0)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss.getData()}")

# Print the summary of the MLP
mlp.summary()

# Test the MLP
print("Testing the trained MLP:")
for inputs in X_data:
    outputs = forward_pass(mlp, inputs)
    print(f"Inputs: {[i.getData() for i in inputs]} -> Output: {[o.getData() for o in outputs]}")
