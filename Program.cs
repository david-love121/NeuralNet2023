
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
            driver.Run();

        }
        
    }
}