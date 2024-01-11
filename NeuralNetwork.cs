using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.PortableExecutable;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;
using System.Text.Json;

namespace NeuralNet2023
{
    
    public class NeuralNetwork
    {
        VectorFunctions vf;
        internal List<Layer> layers;
        Layer firstLayer;
        //Including input layer and output layer
        int[] layerSizes = { 4, 5, 3};
        string[] activationFunctions = { "ReLU", "ReLU", "None" };
        double[]? weights;
        double[]? output;
        internal NeuralNetwork() 
        {
            firstLayer = new Layer();
            layers = new List<Layer>();
            GenerateLayers();
            vf = new VectorFunctions();
        }
        //Used for constructing a new NeuralNetwork object from memory
        internal NeuralNetwork(NeuralNetwork original)
        {
            this.vf = new VectorFunctions();
            layers = new List<Layer>();
            GenerateLayers(original.layers);
            this.firstLayer = layers[0];
            this.layerSizes = original.layerSizes;
            this.activationFunctions = original.activationFunctions;
            this.weights = original.weights;
            this.output = original.output;
        }
        //Used for constructing a new Neuralnetwork object from storage
        internal NeuralNetwork(NeuralNetworkMetadata neuralNetworkMetadata)
        {
            this.vf = new VectorFunctions();
            layers = new List<Layer>();
            firstLayer = new Layer();
            weights = neuralNetworkMetadata.weights;
            layerSizes = neuralNetworkMetadata.layerSizes;
            activationFunctions= neuralNetworkMetadata.activationFunctions;
            GenerateLayers();
        }

        internal void SetFirst(double[] inputs)
        {
            List<Neuron> firstNeurons = firstLayer.GetNeurons();
            for (int i = 0; i < firstNeurons.Count; i++)
            {
                firstNeurons[i].AddInput(inputs[i]);
            }
        }
        internal double[] RunData(double[] inputs)
        {
            List<Neuron> firstNeurons = firstLayer.GetNeurons();
            for (int i = 0; i < firstNeurons.Count; i++)
            {
                firstNeurons[i].AddInput(inputs[i]);
            }
            firstLayer.RunLayersRecursive();
            List<Neuron> outputNeurons = layers.Last().GetNeurons();
            double[] finalLogits = new double[outputNeurons.Count];
            for (int i = 0; i < outputNeurons.Count; i++)
            {
                finalLogits[i] = outputNeurons[i].RunNeuron();
            }
            double[] output = vf.runSoftmax(finalLogits);
            Reset();
            SetOutputLayer(output);
            return output;
        }
        private void GenerateLayers()
        {
            Layer lastLayer = firstLayer;
            int count = 0;
            for (int i = 0; i < layerSizes.Count(); i++)
            {
                Layer layer = new Layer();
                for (int k = 0; k < layerSizes[i]; k++)
                {
                    Neuron neuron = new Neuron(new ActivationFunction(activationFunctions[i]));
                    layer.AddNeuron(neuron);
                    
                }
                layers.Add(layer);
                lastLayer.SetNextLayer(layer);
                List<Neuron> lastNeurons = lastLayer.GetNeurons();
                List<Neuron> currentNeurons = layer.GetNeurons(); 
                for (int countLast = 0; countLast < lastNeurons.Count; countLast++)
                {
                    for (int countCurrent = 0; countCurrent < currentNeurons.Count; countCurrent++)
                    {
                        Connector connector;
                        if (weights == null)
                        {
                            connector = new Connector();
                        }
                        else
                        {
                            connector = new Connector(weights[count]);
                        }
                        connector.SetFirstNeuron(lastNeurons[countLast], countLast);
                        connector.SetSecondNeuron(currentNeurons[countCurrent], countCurrent);
                        layer.AddConnector(connector);
                        count++;
                    }
                }
                lastLayer = layer;
            }
            firstLayer = layers[0];
            lastLayer.SetLast(true);
            return;
        }
       
        //This overload is used when you are generating from a neuralNetwork that already exists

        private void GenerateLayers(List<Layer> original)
        {
            Layer? lastLayer = null;
            for (int i = 0; i < original.Count; i++)
            {
                Layer layer = new Layer(original[i]);
                if (lastLayer != null)
                {
                    lastLayer.SetNextLayer(layer);
                    layer.AttachConnectors(lastLayer, original[i].GetConnectors());
                }
                layers.Add(layer);
                lastLayer = layer;

            }
            return;
        }
        internal void Reset()
        {
            foreach (Layer l in layers)
            {
                foreach (Neuron n in l.GetNeurons())
                {
                    n.ResetNeuron();
                }
            }
            output = null;
        }
        internal double[] GetOutput()
        {
            return output;
        }
        internal void RandomizeWeights()
        {
            foreach (Layer l in layers)
            {
                l.RandomizeWeights();
            }
        }
        internal int[] GetLayerSizes()
        {
            return layerSizes;
        }
        internal string[] GetActivationFunctions()
        {
            return activationFunctions;
        }
        internal List<Layer> GetLayers()
        {
            return layers;
        }
        private void SetOutputLayer(double[] outputValues)
        {
            output = new double[layers.Last().GetNeurons().Count];
            if (output.Length != outputValues.Length)
            {
                throw new ArgumentException("The new values must match the length of the final layer!");
            }
            for (int i = 0; i < output.Length; i++)
            {
                output[i] = outputValues[i];
            }
            return;
        }
        internal double[] GetWeightsArray()
        {
            List<double> weightsList = new List<double>();
            foreach (Layer layer in layers)
            {
                foreach (Connector connector in layer.GetConnectors())
                {
                    weightsList.Add(connector.GetWeight());
                }
            }
            double[] finalWeights = weightsList.ToArray();
            return finalWeights;
        }
        //Only works properly with public getters and setters, need to find a way 
        //To just save metadata to disk
        internal void SaveToStorage(string filePath)
        {
            NeuralNetworkMetadata metadata = new NeuralNetworkMetadata(this);
            string outputString = JsonSerializer.Serialize(metadata);
            NeuralNetworkMetadata metadata2 = JsonSerializer.Deserialize<NeuralNetworkMetadata>(outputString);
            using (StreamWriter outputFile = new StreamWriter(Path.Combine(filePath)))
            {
                outputFile.WriteLine(outputString); 
            }
        }
        internal static NeuralNetwork LoadObjectFromStorage(string filePath)
        {
            string json;
            using (StreamReader reader = new StreamReader(filePath))
            {
                json = reader.ReadToEnd();
            }
            NeuralNetworkMetadata metadata = JsonSerializer.Deserialize<NeuralNetworkMetadata>(json);
            NeuralNetwork finalNetwork = new NeuralNetwork(metadata);
            return finalNetwork;
        }
    }
}
