using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet2023
{
    //Slight misnomer, this should just be called Function as it now contains definitions for more than the activation functions,
    //But I will keep it like this for now
    internal class ActivationFunction
    {
        string selectedFunction = "ReLU";
        string[] types = { "ReLU", "Leaky_ReLU", "Sigmoid", "Sin", "Linear"};
        int coefficient;
        //intercept for use with linear
        int b;
        public ActivationFunction(string t, int coefficient, int b = 0)
        {
            selectedFunction = t;
            this.coefficient = coefficient;
            this.b = b;
        }
        public ActivationFunction(string t)
        {
            selectedFunction = t;
            coefficient = 1;
        }
        public ActivationFunction()
        {
            selectedFunction = "None";
        }
        internal ActivationFunction(ActivationFunction original)
        {
            this.selectedFunction = original.selectedFunction;
            this.types = original.types;
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
            if (selectedFunction == "Sin")
            {
                return Math.Sin(input) * coefficient;
            }
            if (selectedFunction == "Linear")
            {
                return input * coefficient + b;
            }
            return input;
        }

    }
}
