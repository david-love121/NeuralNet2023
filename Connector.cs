using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
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
        internal Connector(double weight)
        {
            this.weight = weight;
        }
        internal void RunData()
        {
            double value = firstNeuron.RunNeuron();
            secondNeuron.AddInput(value * weight);
        }
        
        internal void SetFirstNeuron(Neuron neuron)
        {
            firstNeuron = neuron;
        }
        internal void SetSecondNeuron(Neuron neuron)
        {
            secondNeuron = neuron;
        }
        internal Neuron GetFirstNeuron()
        {
            return firstNeuron;
        }
        internal Neuron GetSecondNeuron()
        {
            return secondNeuron;
        }
        internal void SetWeight(double weight)
        {
            this.weight = weight;
        }
        internal double GetWeight()
        {
            return weight;
        }
    }
}
