
namespace NeuralNet2023 {
    class Program {  
        static void Main(string[] args) {
            TestBackprop();
            
        }
        static void TrainNew(int numTests, bool saveToStorage)
        {
            Driver driver = new Driver();
            driver.TrainEvolutionBased(numTests, saveToStorage);
            driver.Test(1);
        }
        static void LoadFromStorage()
        {
            string path = "C:\\Users\\David\\source\\repos\\NeuralNet2023\\lastNetwork.json";
            Driver driver = new Driver(path);
            double[] check2 = driver.Run();
            driver.Test(50);
        }
        static void TestBackprop()
        {
            string path = "C:\\Users\\David\\source\\repos\\NeuralNet2023\\lastNetwork.json";
            Driver driver = new Driver(path);
            driver.TrainBackpropogationBased(1, false);
        }
    }
}