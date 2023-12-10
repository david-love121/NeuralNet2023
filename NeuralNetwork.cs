using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace NeuralNet2023
{
    public class NeuralNetwork
    {
        VectorFunctions vf;
        List<Layer> layers;
        Layer firstLayer;
        //Including input layer and output layer
        int[] layerSizes = { 4, 5, 3};
        string[] activationFunctions = { "ReLU", "ReLU", "None" };
   
        internal NeuralNetwork() 
        {
            firstLayer = new Layer();
            layers = new List<Layer>();
            GenerateLayers();
            double[] testData = { 0.5, 0.3, 0.3, 0.7, 0.1 };
            vf = new VectorFunctions();

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
            return output;
        }
        private void GenerateLayers()
        {
            Layer lastLayer = firstLayer; 
            for (int i = 0; i < layerSizes.Count(); i++)
            {
                Layer layer = new Layer();
                for (int k = 0; k < layerSizes[i]; k++)
                {
                    Neuron neuron = new Neuron(new ActivationFunction(activationFunctions[i]));
                    layer.AddNeuron(neuron);
                    
                }
                layers.Add(layer);
                layer.AttachLayer(lastLayer);
                List<Neuron> lastNeurons = lastLayer.GetNeurons();
                foreach (Neuron n in lastNeurons)
                {
                    foreach (Neuron n2 in layer.GetNeurons())
                    {
                        Connector connector = new Connector();
                        connector.SetFirstNeuron(n);
                        connector.SetSecondNeuron(n2);
                        layer.AddConnector(connector);
                    }
                }
                lastLayer = layer;
            }
            firstLayer = layers[0];
            lastLayer.SetLast(true);
            return;
        } 
        internal void RandomizeWeights()
        {
            foreach (Layer l in layers)
            {
                l.RandomizeWeights();
            }
        }
        internal void Reset()
        {
            foreach (Layer l in layers)
            {
                foreach (Neuron n in l.GetNeurons())
                {
                    n.resetNeuron();
                }
            }
        }
        internal void SaveToStorage(string filePath)
        {
            XmlSerializer serializer = new XmlSerializer(typeof(NeuralNetwork));
            using (Stream stream = new FileStream(filePath, FileMode.Create, FileAccess.Write))
            {
                serializer.Serialize(stream, this);
            }
        }
        static NeuralNetwork LoadObjectFromStorage(string filePath)
        {
            XmlSerializer serializer = new XmlSerializer(typeof(Neuron));
            using (FileStream stream = new FileStream(filePath, FileMode.Open))
            {
                return (NeuralNetwork)serializer.Deserialize(stream);
            }
        }
    }
}
