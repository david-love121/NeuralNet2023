using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet2023
{
    internal class NeuralNetwork
    {
        VectorFunctions vf;
        public List<Layer> layers;
        Layer firstLayer;
        //Including input layer and output layer
        int[] layerSizes = { 5, 5, 3};
        string[] activationFunctions = { "ReLU", "ReLU", "None" };
   
        internal NeuralNetwork() 
        {
            firstLayer = new Layer();
            layers = new List<Layer>();
            GenerateLayers();
            vf = new VectorFunctions();

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
            foreach (Neuron neuron in outputNeurons)
            {
                //Probably should be neuron.RunNeuron
                finalLogits.Append(neuron.RunNeuron());
            }
            double[] output = vf.runSoftmax(finalLogits);
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
                    layer.addNeuron(neuron);
                    
                }
                layers.Add(layer);
                layer.attachLayer(lastLayer);
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
            lastLayer.setLast(true);
            return;
        } 
    }
}
