using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet2023
{
    internal class Connector
    {
        Neuron firstNeuron;
        Neuron secondNeuron;
        double weight;
        internal Connector()
        {
            weight = 1.0;
        }
        internal void RunData()
        {
            double value = firstNeuron.RunNeuron() * weight;
            secondNeuron.AddInput(value);
        }
        
        internal void SetFirstNeuron(Neuron neuron)
        {
            firstNeuron = neuron;
        }
        internal void SetSecondNeuron(Neuron neuron)
        {
            secondNeuron = neuron;
        }
        internal Neuron GetFirstNeuron(Neuron neuron)
        {
            return firstNeuron;
        }
        internal Neuron GetSecondNeuron(Neuron neuron)
        {
            return secondNeuron;
        }
        internal void SetWeight(double weight)
        {
            this.weight = weight;
        }
    }
}
