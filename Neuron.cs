

namespace NeuralNet2023
{
    internal class Neuron
    {
        ActivationFunction activationFunction;
 
        List<double> inputs;
        List<Connector> connectors;
        public Neuron()
        {
            activationFunction = new ActivationFunction(); 
            inputs = new List<double>();
        }
        public Neuron(ActivationFunction activationFunction)
        {
            this.activationFunction = activationFunction;
            inputs = new List<double>();
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
            inputs.Add(value);
        }

        internal List<Connector> getInputConnector()
        {
            return connectors;
        }

    }
}