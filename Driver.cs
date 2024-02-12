using System;
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
        List<string> guesses;
        List<bool> guessesBool;
        internal Driver()
        {
            dataReader = new DataReader();
            neuralNetwork = new NeuralNetwork();
            guesses = new List<string>(); 
            guessesBool = new List<bool>();
        }
        internal Driver(string objectPath)
        {
            dataReader = new DataReader();
            neuralNetwork = NeuralNetwork.LoadObjectFromStorage(objectPath);
            guesses = new List<string>();
            guessesBool = new List<bool>();
        }
        internal void Test(int numTests)
        {
            string[] answers = dataReader.GetAnswers();
            string[] classifications = dataReader.GetClassifications();
            int countCorrect = 0;
            int countTotal = 0;
            Random random = ManagedRandom.getRandom();
            for (int i = 0; i < numTests; i++)
            {
                countTotal++;
                int rowInd = (int)random.NextInt64(150);
                double[] output = neuralNetwork.RunData(dataReader.GetRow(rowInd));
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
                if (answers[rowInd] == classifications[indHighest])
                {
                    countCorrect++;
                }
                Console.WriteLine($"Test {i}: Running row {rowInd} Prediction: {classifications[indHighest]} Answer: {answers[rowInd]}");
                
            }
            Console.WriteLine($"Test Complete: {countCorrect} Tested, {countTotal} Correct");
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
        internal void TrainBackpropagationBased(int tests, int batchSize, int epochs, bool saveToStorage, double learningRate)
        {
            Random random = ManagedRandom.getRandom();
            List<Layer> layers = neuralNetwork.GetLayers();
            int countLayers = layers.Count;
            int countWeightMatrices = layers.Count - 1;
            //int row = random.Next(dataReader.Height - batchSize);
            List<List<Matrix<double>>> finalAdjustments = new List<List<Matrix<double>>>();
            //weightDerivatives represents a list of Matrices that can be used to update the weights.
            //weight + (derivative * trainingRate) = newWeight
            for (int k = 0; k < countWeightMatrices - 1; k++)
            {
                finalAdjustments.Add(new List<Matrix<double>>());
            }
            finalAdjustments.Add(new List<Matrix<double>>());
            //Memory intensive operation
            for (int a = 0; a < epochs; a++)
            {
                for (int i = 0; i < tests; i++)
                {
                    List<Matrix<double>> allNewWeights = new List<Matrix<double>>();
                    List<Matrix<double>> weightDerivatives = new List<Matrix<double>>();
                    for (int k = 0; k < batchSize; k++)
                    {
                        //Returns nothing but has a side effect on weightDerivatives
                        BackpropogateInitialRun(k, ref weightDerivatives);
                    }
                    //Multiply derivatives by old weight and training rate
                    for (int k = 0; k < weightDerivatives.Count; k++)
                    {
                        int ind = k % countWeightMatrices;
                        double[,] weightsArr = layers[countWeightMatrices - ind].GetWeightsMatrix(layers[countWeightMatrices - ind - 1]);
                        Matrix<double> weights = Matrix<double>.Build.DenseOfArray(weightsArr);
                        Matrix<double> batchChanges = CalculateNewWeightMatrix(weights, weightDerivatives[k], learningRate);
                        allNewWeights.Add(batchChanges);
                    }
                    List<List<Matrix<double>>> seperatedMatrices = new List<List<Matrix<double>>>();
                    for (int k = 0; k < countWeightMatrices; k++)
                    {
                        seperatedMatrices.Add(new List<Matrix<double>>());

                    }
                    for (int k = 0; k < allNewWeights.Count; k++)
                    {
                        int listIndex = k % countWeightMatrices;
                        seperatedMatrices[listIndex].Add(allNewWeights[listIndex]);
                    }
                    //Add all adjustments for each epoch into a list
                    for (int k = 0; k < countWeightMatrices; k++)
                    {
                        int listIndex = k % countWeightMatrices;
                        List<Matrix<double>> currentList = seperatedMatrices[k];
                        Matrix<double> newWeightMatrix = AverageWeightMatrices(currentList);
                        finalAdjustments[listIndex].Add(newWeightMatrix);
                    }

                }
                //Make final adjustments
                for (int k = 0; k < finalAdjustments.Count; k++)
                {
                    List<Matrix<double>> adjustmentsList = finalAdjustments[finalAdjustments.Count - 1 - k];
                    Matrix<double> adjustment = AverageWeightMatrices(adjustmentsList);
                    double[,] weightsArr = adjustment.ToArray();
                    //Check that indexes are matching here?
                    layers[k + 1].UpdateWeights(weightsArr);
                }
            }
        }
        internal Matrix<double> CalculateNewWeightMatrix(Matrix<double> oldWeights, Matrix<double> weightDerivatives, double trainingRate)
        {
            weightDerivatives = weightDerivatives.Multiply(trainingRate);
            Matrix<double> newWeights = oldWeights + weightDerivatives;
            // Matrix<double> newWeights = oldWeights + weightDerivatives;

            return newWeights;
        }
        internal Matrix<double> AverageWeightMatrices(List<Matrix<double>> matrices)
        {
            int rows = matrices[0].RowCount;
            int columns = matrices[0].ColumnCount;
            //Matrices must match in row and column count to average
            foreach (Matrix<double> matrix in matrices)
            {
                if (matrix.RowCount != rows || matrices[0].ColumnCount != columns)
                {
                    throw new Exception("Row count and column count must match");
                }
            }
            Matrix<double> averagedMatrix = Matrix<double>.Build.Dense(rows, columns);
            for (int i = 0; i < matrices.Count; i++)
            {
                averagedMatrix = averagedMatrix + matrices[i];
            }
            averagedMatrix = averagedMatrix.Divide(matrices.Count);
            return averagedMatrix;
        }
        internal void BackpropogateInitialRun(int rowInd, ref List<Matrix<double>> newWeightsStorage)
        {
            (double[] resultsArray, double[] hotCodedArray) = RunSingular(rowInd);
            Vector<double> results = Vector<double>.Build.DenseOfArray(resultsArray);
            Vector<double> hotCoded = Vector<double>.Build.DenseOfArray(hotCodedArray);
            //Where a is a neuron's value post activation, n the value pre activation, w the weight, b the bias
            //L after a variable indicates its a vector of a layer's values, _ means index shift backwards
            List<Layer> layers = neuralNetwork.GetLayers();
            int finalIndex = layers.Count;
            Layer currentLayer = layers[finalIndex - 1];
            Vector<double> zL = Vector<double>.Build.DenseOfArray(currentLayer.GetNeuronPreValues());
            double[,] weightsArr = currentLayer.GetWeightsMatrix(layers[finalIndex - 2]);
            Matrix<double> wL = Matrix<double>.Build.DenseOfArray(weightsArr);
            double bL = currentLayer.GetBias();
            Vector<double> dadz = DerivativeReLU(zL);
            Vector<double> dcda = results - hotCoded;
            Vector<double> chain = dadz.PointwiseMultiply(dcda);
            Vector<double> a_1L = Vector<double>.Build.DenseOfArray(layers[finalIndex - 2].GetNeuronValues());
            List<Vector<double>> columnVector = new List<Vector<double>>();
            //Finds the new weights
            for (int i = 0; i < a_1L.Count; i++)
            {
                //The weights belonging to a single a_1
                double dzdw = a_1L[i];
                Vector<double> newWeights = chain.Multiply(dzdw);
                columnVector.Add(newWeights);
            }
            Matrix<double> newWeightsL = Matrix<double>.Build.DenseOfColumnVectors(columnVector);
            newWeightsL = newWeightsL.Transpose();
            double[] previousda = new double[a_1L.Count];
            int count = 0;
            //Finds the derivatives of a_1 - potentially problematic a_1 is very low
            for (int i = 0; i < a_1L.Count; i++)
            {
                double a_1 = a_1L[i];
                Vector<double> weights = wL.Row(i);
                Vector<double> derivatives = chain.PointwiseMultiply(weights);
                double da_1 = derivatives.Sum();
                previousda[i] = da_1;
            }
            Vector<double> dcda_1L = Vector<double>.Build.DenseOfArray(previousda);

            newWeightsStorage.Add(newWeightsL);
            //Continue to interate
            Backpropagate(dcda_1L, finalIndex - 2, ref newWeightsStorage);
        }
        
        private void Backpropagate(Vector<double> chain, int currentLayerInd, ref List<Matrix<double>> newWeightsStorage) {
            List<Layer> layers = neuralNetwork.GetLayers();
            Layer currentLayer = layers[currentLayerInd];
            Vector<double> zL = Vector<double>.Build.DenseOfArray(currentLayer.GetNeuronPreValues());
            double[,] weightsArr = currentLayer.GetWeightsMatrix(layers[currentLayerInd - 1]);
            Matrix<double> wL = Matrix<double>.Build.DenseOfArray(weightsArr);
            double bL = currentLayer.GetBias();
            Vector<double> dadz = DerivativeReLU(zL);
            chain = dadz.PointwiseMultiply(chain);
            Vector<double> a_1L = Vector<double>.Build.DenseOfArray(layers[currentLayerInd - 1].GetNeuronValues());
            List<Vector<double>> columnVector = new List<Vector<double>>();
            //Finds the new weights
            for (int i = 0; i < a_1L.Count; i++)
            {
                //The weights belonging to a single a_1
                double dzdw = a_1L[i];
                Vector<double> newWeights = chain.Multiply(dzdw);
                columnVector.Add(newWeights);
            }
            Matrix<double> newWeightsL = Matrix<double>.Build.DenseOfColumnVectors(columnVector);
            newWeightsL = newWeightsL.Transpose();
            double[] previousda = new double[a_1L.Count];
            //Finds the derivatives of a_1
            for (int i = 0; i < a_1L.Count; i++)
            {
                double a_1 = a_1L[i];
                Vector<double> weights = wL.Row(i);
                Vector<double> derivatives = chain.PointwiseMultiply(weights);
                double da_1 = derivatives.Sum();
                previousda[i] = da_1;
            }
            Vector<double> dcda_1L = Vector<double>.Build.DenseOfArray(previousda);
            newWeightsStorage.Add(newWeightsL);
            if (currentLayerInd > 1)
            { 
                Backpropagate(dcda_1L, currentLayerInd - 1, ref newWeightsStorage);
            } else
            {
                return;
            }
        }
        
        static internal Vector<double> DerivativeReLU(Vector<double> value)
        {
            Vector<double> result = value.Map(value => value > 0 ? 1.0 : 0.1);
            return result;
        }
        internal double[] Run()
        {
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
