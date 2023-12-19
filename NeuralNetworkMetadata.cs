using System.Text.Json.Serialization;

namespace NeuralNet2023
{
    [Serializable]
    public class NeuralNetworkMetadata
    {
        public int[] layerSizes { get; set; }
        public string[] activationFunctions { get; set; }
        public double[] weights { get; set; }
        public NeuralNetworkMetadata(NeuralNetwork neuralNetwork) 
        { 
            layerSizes = neuralNetwork.GetLayerSizes();
            activationFunctions = neuralNetwork.GetActivationFunctions();
            weights = neuralNetwork.GetWeightsArray();
        }
        [JsonConstructor]
        public NeuralNetworkMetadata(int[] layerSizes, string[] activationFunctions, double[] weights)
        {
            this.layerSizes = layerSizes;
            this.activationFunctions = activationFunctions;
            this.weights = weights;
        }

    }
}