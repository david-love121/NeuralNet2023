﻿using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

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
        
        internal void TrainEvolutionBased(int numIterations, bool saveToStorage)
        {
            double highestScore = 0.0;
            double score;
            for (int i = 0; i < numIterations; i++) 
            {
                double[] results = Run();
                score = results[1] / results[0];
                if (score > highestScore)
                {
                    highestScore = score;
                    Console.WriteLine($"New best score: {highestScore}");
                    bestNetwork = new NeuralNetwork(neuralNetwork);
                }
                neuralNetwork.RandomizeWeights();
            }
            if (saveToStorage)
            {
                bestNetwork.SaveToStorage("C:\\Users\\David\\source\\repos\\NeuralNet2023\\lastNetwork.json");
            }
            //NeuralNetwork checkNetwork = NeuralNetwork.LoadObjectFromStorage("C:\\Users\\David\\source\\repos\\NeuralNet2023\\lastNetwork.xml");
            //bool check = ReferenceEquals(checkNetwork, bestNetwork);
            neuralNetwork = new NeuralNetwork(bestNetwork);
        }
        internal void TrainBackpropogationBased(int epochs, bool saveToStorage)
        {
            Random random = new Random();
            int row = random.Next(dataReader.Height);
            (double[] resultsArray, double[] hotCodedArray) = RunSingular(row);
            Vector<double> results = Vector<double>.Build.DenseOfArray(resultsArray);
            Vector<double> hotCoded = Vector<double>.Build.DenseOfArray(hotCodedArray);
            //Where a is a neuron's value post activation, n the value pre activation, w the weight, b the bias
            //L indicates a layer and after a variable indicates its a vector of a layer's values (unless it's a standalone value)
            List<Layer> layers = neuralNetwork.GetLayers();
            int finalIndex = layers.Count;
            Layer currentLayer = layers[finalIndex - 1];
            Vector<double> zL = Vector<double>.Build.DenseOfArray(currentLayer.GetNeuronPreValues());
            Vector<double> aL = Vector<double>.Build.DenseOfArray(currentLayer.GetNeuronValues());
            double[,] weightsArr = currentLayer.GetWeightsMatrix(layers[finalIndex - 2]);
            Matrix<double> wL = Matrix<double>.Build.DenseOfArray(weightsArr);
            double bL = currentLayer.GetBias();
            Vector<double> dadz = DerivativeReLU(zL);
            Vector<double> dcda = results - hotCoded;
            double db = dadz * dcda;
            int x = 2;
        }
        static internal Vector<double> DerivativeReLU(Vector<double> value)
        {
            Vector<double> result = value.Map(value => value > 0 ? 1.0 : 0.0);
            return result;
        }
        internal double[] Run()
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
                output = neuralNetwork.RunData(dataReader.GetRow(i));
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
            }
            double[] finalScore = {totalPredictions, correctPredictions};
            return finalScore;
            //Todo: load data in from DataReader and then push into NeuralNetwork
        }
        //Returns the prediction and the 1 hot coded solution vector
        internal (double[], double[]) RunSingular(int row)
        {
            string[] answers = dataReader.GetAnswers();
            double[] output = neuralNetwork.RunData(dataReader.GetRow(row));
            double[] hotCoded = new double[output.Length];
            double highestProbability = 0;
            int indHighest = 0;
            for (int i = 0; i < output.Length; i++)
            {
                if (output[i] > highestProbability)
                {
                    highestProbability = output[i];
                    indHighest = i;
                }
            }
            string[] classifications = dataReader.GetClassifications();
            for (int k = 0; k < classifications.Length; k++)
            {
                if (answers[row] != classifications[k])
                {
                    hotCoded[k] = 0;
                } else 
                {
                    hotCoded[k] = 1;
                }
            }
            
            return (output, hotCoded);

        }
    }
}
