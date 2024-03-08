using System.Text.Json.Serialization;

namespace NeuralNet2023
{
    [Serializable]
    public class NeuralNetworkMetadata
    {
        public int[] layerSizes { get; set; }
        public string[] activationFunctions { get; set; }
        public double[] weights { get; set; }
        public double[] biases { get; set; }
        public NeuralNetworkMetadata(NeuralNetwork neuralNetwork) 
        { 
            layerSizes = neuralNetwork.GetLayerSizes();
            activationFunctions = neuralNetwork.GetActivationFunctions();
            weights = neuralNetwork.GetWeightsArray();
            biases = neuralNetwork.GetBiasesArray();
        }
        [JsonConstructor]
        public NeuralNetworkMetadata(int[] layerSizes, string[] activationFunctions, double[] weights, double[] biases)
        {
            this.layerSizes = layerSizes;
            this.activationFunctions = activationFunctions;
            this.weights = weights;
            this.biases = biases;
        }

    }
}