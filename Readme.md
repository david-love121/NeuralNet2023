### Structure
The neural network is initalized by creating a first layer with a number of neurons defined by the user, usually equal to the number of parameters in your dataset. Then:
- Creating layers based on the amount of hidden layers.
- Creating neurons for each hidden layer.
- Creating connectors for each of the layers.
- Running the connectors recursively.

Each layer contains neurons and layers run their neurons. Each layer is ran by the layer before it recursively.
Each neuron is defined by, $z^L = w^La^{L-1}+b$
