
namespace NeuralNet2023 {
    class Program {  
        static void Main(string[] args) {
            //Driver driver = new Driver();
            //driver.Train(1000000);
            //double[] output = driver.Run(false);
            //driver.Test(50);
            string path = "C:\\Users\\David\\source\\repos\\NeuralNet2023\\lastNetwork.json";
            Driver driver2 = new Driver(path);
            double[] check2 = driver2.Run();
            driver2.Test(50);
        }
    }
}