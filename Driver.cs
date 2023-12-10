using System;
using System.Collections.Generic;
using System.ComponentModel;
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
        NeuralNetwork bestNetwork;
        internal Driver()
        {
            dataReader = new DataReader();
            neuralNetwork = new NeuralNetwork();
        }
        internal void Train()
        {
            double highestScore = 0.0;
            int numTests = 10;
            for (int i = 0; i < numTests; i++) 
            {
                double[] results = Run();
                double score = results[1] / results[0];
                if (score > highestScore)
                {
                    highestScore = score;
                    bestNetwork = neuralNetwork;
                }
                neuralNetwork.RandomizeWeights();
            }
            bestNetwork.SaveToStorage("./lastNetwork.xml");
            int x = 2;
            
        }
        internal double[] Run()
        {
            double[] testData = { 0.5, 0.3, 0.3, 0.7, 0.1 };
            List<string> allData = dataReader.GetData();
            double[,] features = dataReader.GetFeatures();
            string[] answers = dataReader.GetAnswers();
            neuralNetwork.RandomizeWeights();
            int correctPredictions = 0;
            int totalPredictions = 0;
            for (int i = 0; i < dataReader.Height; i++)
            {
                int indHighest = 0;
                double[] output = neuralNetwork.RunData(dataReader.GetRow(i));
                double[] normalizedOutput = new double[output.Length];
                double highestProbability = 0;
                for (int k = 0; k < output.Length; k++)
                {
                    if (output[k] > highestProbability)
                    {
                        highestProbability = output[k];
                        indHighest = k;
                    }

                }
                normalizedOutput[indHighest] = 1;
                string selectedClass = dataReader.GetClassifications()[indHighest];
                if (selectedClass == answers[i])
                {
                    correctPredictions++;
                }
                totalPredictions++;
            }
            double[] finalScore = {totalPredictions, correctPredictions};
            return finalScore;
            
            
            //Todo: load data in from DataReader and then push into NeuralNetwork
        }
    }
}
