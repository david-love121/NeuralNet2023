namespace NeuralNet2023
{
   
    internal class Layer
    {
        List<Connector> connectors;
        List<Neuron> neurons;
        Layer nextLayer;
        bool last;
        Random random = new Random();
        internal Layer()
        {
            last = false;
            connectors = new List<Connector>();
            neurons = new List<Neuron>();
        }
        internal Layer(bool last)
        {
            this.last = last;
            connectors = new List<Connector>();
            neurons = new List<Neuron>();
        }
        internal Layer(Layer original)
        {
            this.last = original.last;
            this.connectors = new List<Connector>();
            this.neurons = new List<Neuron>();
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
        internal void SetNextLayer(Layer layer)
        {
            this.nextLayer = layer;
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
                connector.SetWeight(random.NextDouble());
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