
using static System.Net.Mime.MediaTypeNames;

namespace NeuralNet2023 {
    class Program {  
        static void Main(string[] args) {
            Driver driver = LoadFromStorage();
            double[] output = driver.RunTestFunction();
            double accuracyPreBackprop = driver.CheckTestFunctionAccuracy();
            driver.TrainBackpropagationBased(1000, 100, 45, false, 1);
            double accuracyPostBackprop = driver.CheckTestFunctionAccuracy();
            driver.TrainBackpropagationBased(1000, 100, 15, false, 1);
            double accuracyPostBackprop2 = driver.CheckTestFunctionAccuracy();
            int x = 2;

        }
        static Driver TrainNew(int numTests, bool saveToStorage)
        {
            Driver driver = new Driver();
            driver.TrainEvolutionBased(numTests, saveToStorage);
            //driver.TrainBackpropagationBased(10000, 200, 4, false, 0.1);
            //driver.Test(50);
            double[] score = driver.RunTestFunction();
            double accuracy = driver.CheckTestFunctionAccuracy();

            return driver;
        }
        static Driver LoadFromStorage()
        {
            string path = "C:\\Users\\David\\source\\repos\\NeuralNet2023\\lastNetwork.json";
            Driver driver = new Driver(path);
            return driver;
            
        }
        static void TestBackprop(Driver driver, double learningRate, int tests, int batchSize)
        {
 
            
        }
    }
}