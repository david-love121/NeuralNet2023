using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet2023
{
    
    internal class DataGenerator
    {
        ActivationFunction function;
        double[] data;
        internal int numPoints { get; private set; }
        internal DataGenerator(int points) 
        {
            function = new ActivationFunction("Sin", 1);
            data = new double[points];
            numPoints = points;
            PopulateData(points);
        }
        internal void PopulateData(int points)
        {
            for (int i = 0; i < points; i++)
            {
                data[i] = function.RunData(i);
            }
        }
        internal double GetDataPoint(int point)
        {
            return data[point];
        }

    }
}
