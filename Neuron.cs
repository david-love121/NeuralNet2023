

namespace NeuralNet2023
{
    internal class Neuron
    {
        ActivationFunction activationFunction;
        double value;
        List<double> inputs;
        List<Connector> connectors;
        List<Connector> nextConnectors;
        public Neuron()
        {
            activationFunction = new ActivationFunction(); 
        }
        public Neuron(ActivationFunction activationFunction)
        {
            this.activationFunction = activationFunction;
        }
        internal double RunNeuron()
        {
            double sum = 0.0;
            foreach (double input in inputs)
            {
                sum = sum + input;
            }
            double nextValue = activationFunction.RunData(sum);
            return nextValue;
        }
        internal double RunNeuron(double input)
        {
            return activationFunction.RunData(input);
        }
        internal void AddInput(double value)
        {
            this.value = value;
        }
        
        internal List<Connector> getInputConnector()
        {
            return connectors;
        }

    }
}