
using System.Reflection.PortableExecutable;
using System.Runtime.CompilerServices;

namespace NeuralNet2023 {
    class Program {
        DataReader dataReader;
        
        public Program()
        {
            
        }
        static void Main(string[] args) {
            Driver driver = new Driver();
            driver.Train();
            double[] output = driver.Run(false);
            driver.Test(50);
            //string path = "C:\\Users\\David\\source\\repos\\NeuralNet2023\\lastNetwork.xml";
            //Driver driver2 = new Driver(path);
            //double[] check2 = driver2.Run(false);
        }
        
    }
}