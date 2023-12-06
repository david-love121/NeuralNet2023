using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet2023
{
    
    internal class Driver
    {
        DataReader dataReader;
        NeuralNetwork neuralNetwork;
        
        internal Driver()
        {
            dataReader = new DataReader();
            neuralNetwork = new NeuralNetwork();
        }
        internal void Train()
        {

        }
        internal void Run()
        {
            double[] testData = { 0.5, 0.3, 0.3, 0.7, 0.1 };
            neuralNetwork.RunData(testData);
            int x = 2;
        }
    }
}
