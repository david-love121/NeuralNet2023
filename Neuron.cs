

namespace NeuralNet2023
{
    internal class Neuron
    {
        ActivationFunction activationFunction;
        List<double> inputs;
        double value;
        internal Neuron()
        {
            activationFunction = new ActivationFunction(); 
            inputs = new List<double>();
        }
        internal Neuron(ActivationFunction activationFunction)
        {
            this.activationFunction = activationFunction;
            inputs = new List<double>();
        }
        internal Neuron(Neuron original)
        {
            this.activationFunction = new ActivationFunction(original.activationFunction);
            this.inputs = new List<double>();
            foreach (double d in original.inputs)
            {
                inputs.Add(d);
            }
            this.value = original.value;
        }
        internal double RunNeuron()
        {
            double sum = 0.0;
            foreach (double input in inputs)
            {
                sum = sum + input;
            }
            value = activationFunction.RunData(sum);

            return value;
        }
        internal double RunNeuron(double input)
        {
            return activationFunction.RunData(input);
        }
        internal void AddInput(double value)
        {
            inputs.Add(value);
        }

        internal double GetValue()
        {
            return value;
        }
        internal void resetNeuron()
        {
            inputs.Clear();
        }
    }
}