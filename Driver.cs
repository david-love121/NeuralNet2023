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
        internal NeuralNetwork neuralNetwork;
        internal NeuralNetwork bestNetwork;
        List<string> guesses = new List<string>();
        List<bool> guessesBool = new List<bool>();
        internal Driver()
        {
            dataReader = new DataReader();
            neuralNetwork = new NeuralNetwork();
        }
        internal Driver(string objectPath)
        {
            dataReader = new DataReader();
            neuralNetwork = NeuralNetwork.LoadObjectFromStorage(objectPath);
        }
        internal void Test(int numTests)
        {
            double[,] features = dataReader.GetFeatures();
            string[] answers = dataReader.GetAnswers();
            Random random = new Random();
            for (int i = 0; i < numTests; i++)
            {
                int rowInd = (int)random.NextInt64(150);
                double[] output = neuralNetwork.RunData(dataReader.GetRow(rowInd));
                double[] normalizedOutput = new double[output.Length];
                double highestProbability = 0;
                int indHighest = 0;
                for (int k = 0; k < output.Length; k++)
                {
                    if (output[k] > highestProbability)
                    {
                        highestProbability = output[k];
                        indHighest = k;
                    }
                }
                Console.WriteLine($"Test {i}: Running row {rowInd} Prediction: {dataReader.GetClassifications()[indHighest]} Answer: {answers[rowInd]}");
            }
        }
        internal void Train()
        {
            double highestScore = 0.0;
            int numTests = 1000;
            double score;
            for (int i = 0; i < numTests; i++) 
            {
                double[] results = Run(false);
                score = results[1] / results[0];
                if (score > highestScore)
                {
                    highestScore = score;
                    Console.WriteLine($"New best score: {highestScore}");
                    bestNetwork = new NeuralNetwork(neuralNetwork);
                }
                neuralNetwork.RandomizeWeights();
            }
            bestNetwork.SaveToStorage("C:\\Users\\David\\source\\repos\\NeuralNet2023\\lastNetwork.json");
            //NeuralNetwork checkNetwork = NeuralNetwork.LoadObjectFromStorage("C:\\Users\\David\\source\\repos\\NeuralNet2023\\lastNetwork.xml");
            //bool check = ReferenceEquals(checkNetwork, bestNetwork);
            neuralNetwork = new NeuralNetwork(bestNetwork);


        }
        internal double[] Run(bool useBest)
        {
            bool check = ReferenceEquals(bestNetwork, neuralNetwork);
            guesses.Clear();
            guessesBool.Clear();
            //List<string> allData = dataReader.GetData();
            //double[,] features = dataReader.GetFeatures();
            string[] answers = dataReader.GetAnswers();
            int correctPredictions = 0;
            int totalPredictions = 0;
            double[] output;
            for (int i = 0; i < dataReader.Height; i++)
            {
                
                int indHighest = 0;
                if (useBest)
                {
                    output = bestNetwork.RunData(dataReader.GetRow(i));
                }
                else
                {
                    output = neuralNetwork.RunData(dataReader.GetRow(i));
                }
                double highestProbability = 0;
                for (int k = 0; k < output.Length; k++)
                {
                    if (output[k] > highestProbability)
                    {
                        highestProbability = output[k];
                        indHighest = k;
                    }

                }
                string selectedClass = dataReader.GetClassifications()[indHighest];
                if (selectedClass == answers[i])
                {
                    correctPredictions++;
                   
                }
                totalPredictions++;
                guesses.Add(selectedClass);
                guessesBool.Add(selectedClass == answers[i]);
                //reset neural net
            }
            double[] finalScore = {totalPredictions, correctPredictions};
            
            return finalScore;
            
            
            //Todo: load data in from DataReader and then push into NeuralNetwork
        }
    }
}
