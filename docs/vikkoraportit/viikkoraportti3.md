# Week 3

During the third week, I moved from dataset preparation toward the actual neural network implementation. I started building the MLP from scratch and added the core pieces needed for forward propagation and training.

Most of the work went into implementing the model structure in NumPy. I created a `Layer` class for the dense layers, added ReLU and Softmax activation functions, and updated the loss functions so the network can use categorical cross-entropy during training. I also wrote a small demo file to test the forward pass and better understand how the data flows through the network layer by layer.

The biggest challenge was building the network itself. I had to make sure the implementation was simple enough to match the course requirements, but still structured well enough to support training and later improvements. Getting the layers, activations, and loss calculations to work together correctly took careful debugging, especially because small mistakes in matrix shapes can easily break the whole system.

I have not implemented any user interface yet. Next week, I will start working on the UI and move from synthetic digit data toward parsing real receipt data.

## Time Tracking

| Date | Time Spent | Description |
| ----- | ----------- | ----------- |
| 28.5. | 4 h | Implementing the MLP layer structure and forward propagation |
| 29.5. | 6 h | Adding activation functions and updating loss functions |
| 30.5. | 3 h | Creating the demo file and testing the network flow |
| 31.5. | 8 h | Writing and improving generator tests |
| **Total** | **21 h** | |
