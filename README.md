## In place of a more traditional readme while I'm still working, I've just uploaded my notes about the project.
### General Notes
Language of choice is going to be C#. Perhaps remake in C++ if I can get something even working in C#. 
Dataset will be the iris dataset. Free at https://archive.ics.uci.edu/dataset/53/iris
Initialization occurs by:
- Creating a first layer with a number of neurons equal to the number of inputs. 
- Creating layers based on the amount of hidden layers.
- Creating neurons for each hidden layer.
- Creating connectors for each of the layers.
- Running the connectors recursively.
### Code Structure
Main creates a NeuralNetwork object that will activate the rest of the code. Descriptions of each file can be found below.
#### DataReader
DataReader takes in data from a file in the /data subdirectory (must be included in the output, searches in the bin of project.) DataReader then has multiple methods for interpreting and reading the data from file.
#### Neural Network
Neural Network creates the rest of the data elements and interprets them. Neural network runs the first layer and this layer runs the rest of the layers.
#### Layer
Layers will contain a list of connectors. Once the first layer is ran all of the connectors should run each other recursively. Layers are connected to each other and can run their connectors in order. Once the first layer runs its connectors, all the layers will run their connectors.
#### Connector
Connectors contain an output neuron, a list of input neurons, and a weight value. Connectors are responsible for running the data and moving data down the net.  
#### Neuron
Neuron is a singular neuron part of a larger layer. A neuron has data and an activation function. A neuron knows its data and the next connector to send its data through. Neurons run nothing and only return their data.

#### Neural Network Metadata
The NeuralNetworkMetadata object contains only the necessary info about a neural network to save it to disk and reconstruct it using the overloaded constructor. It has internal getter and setters to make saving to disk easy. 

### General Neural Networks
A neural network contains 3 types of components.
- Input layer
- Hidden layer
- Output layer
Inputs are taken into a neuron which then outputs into either another neuron or the final output. Neural networks differ from perceptron as they have multiple layers and multiple neurons.
Neurons take inputs, (x1, x2,) multiply by a weight, apply an activation function, and then pass it on
#### Gradient Descent
Gradient descent describes using derivatives to optimize the cost function for the minimum. This is done by initializing the weights randomly, calculating the error, and then finding the derivative of the error with respect to the weighted input neuron.
W$_x$ = W - a(Gradient)
Where a is a learning rate.
In training you will first use the softmax function to convert your logits to a vector of 3 values. This vector of 3 values will be compared to a one-hot encoded vector that corresponds to the class. You must match the final number of neurons to the number of classes. 
3 logits -> softmax function -> loss function
The loss function is used in training to optimize the network. I will be using categorical cross entropy loss. The derivative of the loss function is equal simply to s - y, where s is the output of the softmax function and y the one-hot encoded vector. The derivative has to be calculated for error with respect to each different waited input. 

Running data through, a potential paradigm is Connector runs neurons and neurons run connectors. First neurons would be built and added into their layers, then connectors would be built between neurons. The first line of neurons are ran, and these neurons run their connectors. (I kept this paradigm except layers are now the originator, not neurons.)

#### Cross Entropy Categorical Loss
Cross Entropy Categorical (CCE) Loss describes a popular of defining loss in a neural network.

#### Backpropagation 
https://www.youtube.com/watch?v=aircAruvnKk
From the 3Blue1Brown series
Start by representing all inputs a column vector. Then, represent all weights as a matrix.
Ex:
z00a00   w00 w01 w0n
z10a10   w10 w11 w1n
z20an0   wn0 wn1 wnn
(While my notes include details for bias, I haven't added this yet.)
$$
z^L=w^La^{L-1}+b
$$
$$
a^L=ReLU(z^L)
$$
Once you do this, you can use the MathNET Numerics library to manipulate these weights easily. 
Backpropagation involves using the chain rule to determine the derivative of the cost function with respect to the weighted inputs. I am using a ReLU activation and Categorical Cross Entropy (CCE.)
$$
\frac{\partial C_0}{\partial W^L} = \frac{\partial z^L}{\partial W^L}\frac{\partial a^L}{\partial z^L}\frac{\partial C_0}{\partial a^L}
$$
where:
$$\frac{\partial z^L}{\partial W^L} =a^{L-1}$$ $$\frac{\partial a^L}{\partial{z^L}} = \begin{cases} 1 & z^L > 0 \\ 0 & z^L < 0 \end{cases}$$
$$
\frac{\partial C_0}{\partial a^L} = {s - y} \text{ where } \mathbf{s} = \begin{bmatrix} s_1 \\ s_2 \\ s_3 \end{bmatrix} \text{ and y is predicted output } 
$$
Note that $C_0$ is only a *singular* training example, and the derivative of $C$ with respect to the weights is equal to the average of all training samples, or:
$$
\frac{\partial C}{\partial w^L} = \frac{1}{n}\sum_{k=0}^{n-1}\frac{\partial C_k}{\partial w^L}
$$
Which is only one dimension of $\nabla C$ . Finding all of $\nabla C$ isn't complicated, as it only requires switching the first term. 
The cost for a given layer with respect to the activations is simply the sum of the cost of each of the neurons, or in our case $s - \hat{y}$. Finally, the derivative of the cost with respect to $a_k^{L-1}$:
$$
\frac{\partial C_0}{\partial a_k^{L-1}} = \sum_{j=0}^{n_L-1}\frac{\partial z_j^L}{\partial a_k^{L-1}}\frac{\partial a^L_j}{\partial z^L_j}\frac{\partial C_0}{\partial a^L_j}
$$
And the derivative of the cost with respect to $w^L_{jk}$:
$$
\frac{\partial C_0}{\partial W^L_{jk}} = \frac{\partial z^L_j}{\partial W^L_{jk}}\frac{\partial a^L_j}{\partial z^L_j}\frac{\partial C_0}{\partial a^L_j}
$$
Where j is the index of the neuron on the current layer, k the index of the previous layer, and L the current layer. This is because one neuron of the previous layer affects each of the next. Once you have calculated the cost of the final layer with respect to the activations of the previous layer, you can **backpropagate** this to the rest of the layers. For each layer, a gradient vector will be generated, denoted by ${\nabla C}$.
$$
{\nabla C} = \begin{bmatrix} \frac{\partial C}{\partial w^1} \\ \frac{\partial C}{\partial b^1} \\ \text{...} \\ \frac{\partial C}{\partial w^L} \\ \frac{\partial C}{\partial b^L} \end{bmatrix}
$$



