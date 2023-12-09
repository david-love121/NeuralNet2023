using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet2023
{
    
    internal class ActivationFunction
    {
        string selectedFunction = "ReLU";
        string[] types = { "ReLU", "Leaky_ReLU", "Sigmoid", "Softmax" };
        public ActivationFunction(string t)
        {
            selectedFunction = t;
        }
        public ActivationFunction()
        {
            selectedFunction = "None";
        }
        public double RunData(double input)
        {
            if (selectedFunction == "ReLU")
            {
                return Math.Max(0, input);
            }
            if (selectedFunction == "Leaky_ReLU")
            {
                return Math.Max(input * 0.01, input);
            }
            if (selectedFunction == "Sigmoid")
            {
                return 1.0 / (1.0 + Math.Exp(-input));
            }
            return input;
        }

    }
}
