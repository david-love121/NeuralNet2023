using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
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
            double trainingRate = 1;
            
        }
        internal void Run()
        {
            double[] testData = { 0.5, 0.3, 0.3, 0.7, 0.1 };
            double[] output = neuralNetwork.RunData(testData);
            double x = 2.0;
            //Todo: load data in from DataReader and then push into NeuralNetwork
        }
    }
}
