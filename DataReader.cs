using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.ConstrainedExecution;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet2023
{
    internal class DataReader
    {
        List<String> finalData = new List<String>();
        string currentDirectory = Directory.GetCurrentDirectory();
        string dataPath = "./data/iris.data";
        readonly string[] names = { "sepalLength", "septalWidth", "petalLength", "petalWidth", "classification" };
        readonly string[] classifications = {"Iris-setosa", "Iris-veriscolour", "Iris-virginica"};
        public DataReader()
        {
            string filePath = Path.Combine(currentDirectory, dataPath);
            if (!File.Exists(filePath))
            {
                finalData.Add("Data not found.");
                return;
                
            }
            using (StreamReader reader = new StreamReader(filePath)) 
            { 
                while (!reader.EndOfStream)
                {
                    string line = reader.ReadLine();
                    finalData.Add(line);
                }
            }
            return;
            
        }
        public List<String> GetData()
        {
            return finalData;
        }
        public string[] GetNames()
        {
            return names;
        }
        public string[] GetClassifications()
        {
            return classifications;
        }
    }
}
