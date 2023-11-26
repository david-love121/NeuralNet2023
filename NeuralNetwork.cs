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
        DataReader dataReader = new DataReader();
        public List<Layer> layers = new List<Layer>();
        Layer firstLayer;
        int[] layerSizes = { 5, 5 };
   
        internal NeuralNetwork() 
        {
            firstLayer = new Layer();
            GenerateLayers();
            int x = 2;
            
        }
        void Train()
        {

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
            Layer finalLayer = new Layer(true, lastLayer);
            
            return;
        } 
    }
}
