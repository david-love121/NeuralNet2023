namespace NeuralNet2023
{
   
    internal class Layer
    {
        public List<Connector> connectors;
        public List<Neuron> neurons;
        Layer nextLayer;
        bool last;
        internal Layer()
        {
            last = false;
            connectors = new List<Connector>();
            neurons = new List<Neuron>();
        }
        internal Layer(bool last, Layer nextLayer)
        {
            this.last = last;
            connectors = new List<Connector>();
            neurons = new List<Neuron>();
            this.nextLayer = nextLayer; 
        }
        internal void RunLayer(List<double> inputs)
        {
            if (inputs.Count != connectors.Count)
            {
                throw new Exception("Number of inputs does not match neurons.");
            }
            for (int i = 0; i < connectors.Count; i++)
            {
                connectors[i].RunData();
            }
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
        internal void attachLayer(Layer layer)
        {
            
            this.nextLayer = layer;
        }
        internal List<Neuron> GetNeurons()
        {
            return neurons;
        }
        internal void addNeuron(Neuron neuron)
        {
            neurons.Add(neuron);
        }
    }
}