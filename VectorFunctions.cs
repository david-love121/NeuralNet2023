using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet2023
{
    internal class VectorFunctions
    {
        public double[] runSoftmax(double[] inputs)
        {
            double[] outputs = new double[inputs.Length];
            double sum = 0;
            for (int i = 0; i < inputs.Count(); i++)
            {
                //e^input
                sum = sum + Math.Exp(inputs[i]);
            }
            for (int i = 0; i < inputs.Count(); i++)
            {
                outputs[i] = Math.Exp(inputs[i]) / sum;
            }
            return outputs;
        }
    }
}
