
namespace NeuralNet2023 {
    class Program {  
        static void Main(string[] args) {
            //TrainNew(30000, true);
            TestBackprop(2, 1000, 150);
            
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
            double[] check2 = driver.Run();
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