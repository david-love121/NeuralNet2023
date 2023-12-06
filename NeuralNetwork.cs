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

        public List<Layer> layers;
        Layer firstLayer;
        //Including input and output layer
        int[] layerSizes = { 5, 5, 5, 5, 1};
   
        internal NeuralNetwork() 
        {
            firstLayer = new Layer();
            layers = new List<Layer>();
            GenerateLayers();
            int x = 2;
            
        }
        internal void RunData(double[] inputs)
        {
            List<Neuron> firstNeurons = firstLayer.GetNeurons();
            for (int i = 0; i < firstNeurons.Count; i++)
            {
                firstNeurons[i].AddInput(inputs[i]);
            }
            firstLayer.RunLayersRecursive();
            

        }
        private void GenerateLayers()
        {
            Layer lastLayer = firstLayer; 
            for (int i = 0; i < layerSizes.Count(); i++)
            {
                Layer layer = new Layer();
                for (int k = 0; k < layerSizes[i]; k++)
                {
                    Neuron neuron = new Neuron(new ActivationFunction("ReLU"));
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
