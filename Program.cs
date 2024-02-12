
namespace NeuralNet2023 {
    class Program {  
        static void Main(string[] args) {
            LoadFromStorage();
            
            
        }
        static void TrainNew(int numTests, bool saveToStorage)
        {
            Driver driver = new Driver();
            driver.TrainEvolutionBased(numTests, saveToStorage);
            driver.Test(50);
        }
        static void LoadFromStorage()
        {
            string path = "C:\\Users\\David\\source\\repos\\NeuralNet2023\\lastNetwork.json";
            Driver driver = new Driver(path);
            TestBackprop(0.1, 10000, 150);
            driver.Test(50);
        }
        static void TestBackprop(double learningRate, int epochs, int batchSize)
        {
            string path = "C:\\Users\\David\\source\\repos\\NeuralNet2023\\lastNetwork.json";
            Driver driver = new Driver(path);
            driver.TrainBackpropagationBased(epochs, batchSize, false, learningRate);
            driver.Test(50);
        }
    }
}