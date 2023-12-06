namespace NeuralNet2023
{
   
    internal class Layer
    {
        List<Connector> connectors;
        List<Neuron> neurons;
        Layer nextLayer;
        bool last; 

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
        internal void attachLayer(Layer layer)
        {

            layer.SetNextLayer(this);
        }
        internal List<Neuron> GetNeurons()
        {
            return neurons;
        }
        internal void addNeuron(Neuron neuron)
        {
            neurons.Add(neuron);
        }
        internal void setLast(bool last)
        {
            this.last = last;   
        }
    }
}