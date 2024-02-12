
using static System.Net.Mime.MediaTypeNames;

namespace NeuralNet2023 {
    class Program {  
        static void Main(string[] args) {
            Driver driver = TrainNew(10000, false);
            driver.TrainBackpropagationBased(10000, 150, 3, false, 0.1);
            driver.Test(149);


        }
        static Driver TrainNew(int numTests, bool saveToStorage)
        {
            Driver driver = new Driver();
            driver.TrainEvolutionBased(numTests, saveToStorage);
            driver.Test(50);
            return driver;
        }
        static void LoadFromStorage()
        {
            string path = "C:\\Users\\David\\source\\repos\\NeuralNet2023\\lastNetwork.json";
            Driver driver = new Driver(path);
            
            driver.Test(50);
        }
        static void TestBackprop(Driver driver, double learningRate, int tests, int batchSize)
        {
 
            
        }
    }
}