namespace NeuralNet2023
{
   
    internal class Layer
    {
        List<Connector> connectors;
        List<Neuron> neurons;
        Layer nextLayer;
        bool last;
        double bias;
        Random random = ManagedRandom.getRandom();
        internal Layer()
        {
            last = false;
            connectors = new List<Connector>();
            neurons = new List<Neuron>();
            bias = 0;
        }
        internal Layer(bool last)
        {
            this.last = last;
            connectors = new List<Connector>();
            neurons = new List<Neuron>();
            bias = 0;
        }
        internal Layer(Layer original)
        {
            this.last = original.last;
            this.connectors = new List<Connector>();
            this.neurons = new List<Neuron>();
            this.bias = original.bias;
            foreach (Neuron neuron in original.neurons)
            {
                Neuron newNeuron = new Neuron(neuron);
                this.neurons.Add(newNeuron);
            }
        }
        internal void RunLayer()
        {
            for (int i = 0; i < connectors.Count; i++)
            {
                connectors[i].RunData();
            }
        }
        internal void RunLayersRecursive()
        {
            foreach (Neuron neuron in neurons)
            {
                neuron.AddInput(bias);
            }
            for (int i = 0; i < connectors.Count; i++)
            {
                connectors[i].RunData();
            }
            if (last)
            {
                return;
            }
            nextLayer.RunLayersRecursive();
        }
        internal double[] GetNeuronValues()
        {
            double[] values = new double[neurons.Count];
            for (int i = 0; i < neurons.Count; i++) {
                values[i] = neurons[i].GetValue();
            }
            return values;
        }
        internal double[] GetNeuronPreValues()
        {
            double[] values = new double[neurons.Count];
            for (int i = 0; i < neurons.Count; i++)
            {
                values[i] = neurons[i].GetPreactivationValue();
            }
            return values;
        }
        internal void SetNextLayer(Layer layer)
        {
            this.nextLayer = layer;
        }
        internal double GetBias()
        {
            return bias;
        }
        internal void SetBias(double bias)
        {
            this.bias = bias;
        }
        internal void AddConnector(Connector c)
        {
            connectors.Add(c);
            return;
        }
        internal List<Connector> GetConnectors()
        {
            return connectors;
        }
        internal List<Neuron> GetNeurons()
        {
            return neurons;
        }
        internal void AddNeuron(Neuron neuron)
        {
            neurons.Add(neuron);
        }
        internal void SetLast(bool last)
        {
            this.last = last;   
        }
        internal void RandomizeWeights()
        {
            foreach (Connector connector in connectors)
            {
                connector.SetWeight((random.NextDouble() + 0.05));
            }
        }
        internal double[,] GetWeightsMatrix(Layer lastLayer)
        {
            int count = 0;
            double[,] finalResult = new double[lastLayer.GetNeurons().Count, this.neurons.Count];
            for (int i = 0; i < lastLayer.GetNeurons().Count; i++)
            {
                for (int k = 0; k < this.neurons.Count; k++)
                {
                    finalResult[i, k] = connectors[count].GetWeight();
                    count++;
                }
            }
            return finalResult;
        }
        internal void UpdateWeights(double[,] weights)
        {
            int height = weights.GetLength(0);
            int width = weights.GetLength(1);
            for (int i = 0; i < height; i++)
            {
                for (int k = 0; k < width; k++)
                {
                    connectors[i * k].SetWeight(weights[i, k]);
                }
            }
        }
        internal void AttachConnectors(Layer lastLayer, List<Connector> originalConnectors)
        {
            int currentConnector = 0;
            List<Neuron> lastNeurons = lastLayer.GetNeurons();
            for (int i = 0; i < lastLayer.GetNeurons().Count; i++)
            {
                for (int k = 0; k < neurons.Count; k++)
                {
                    Connector newConnector = new Connector(originalConnectors[currentConnector].GetWeight());
                    newConnector.SetFirstNeuron(lastNeurons[i], i);
                    newConnector.SetSecondNeuron(neurons[k], k);
                    connectors.Add(newConnector);
                    currentConnector++;
                }
            }
        }
    }
}