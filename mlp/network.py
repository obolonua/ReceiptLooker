
import numpy as np

from layers import Layer
from activations import ReLU, Softmax
from losses import CategoricalCrossEntropy


# Sample batch of 5 flattened images with 784 input features each.
X = np.random.randn(5, 784)

# First hidden layer: 784 inputs -> 128 features.
layer1 = Layer(784, 128)
activation1 = ReLU()

# Second hidden layer: 128 features -> 64 features.
layer2 = Layer(128, 64)
activation2 = ReLU()

# Output layer: 64 features -> 10 class scores.
layer3 = Layer(64, 10)
softmax = Softmax()

# Loss function used to compare predicted probabilities with labels.
loss_function = CategoricalCrossEntropy()


# Forward pass
# Run the batch through the network one layer at a time.
output1 = layer1.forward(X)
output2 = activation1.forward(output1)

output3 = layer2.forward(output2)
output4 = activation2.forward(output3)

output5 = layer3.forward(output4)

predictions = softmax.forward(output5)

# Print the predicted class probabilities and their shape.
print(predictions)
print(predictions.shape)
