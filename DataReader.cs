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
        List<string> allData = new List<string>();
        string currentDirectory = Directory.GetCurrentDirectory();
        string dataPath = "./data/iris.data";
        readonly string[] names = { "sepalLength", "septalWidth", "petalLength", "petalWidth", "classification" };
        readonly string[] classifications = { "Iris-setosa", "Iris-versicolor", "Iris-virginica" };
        double[,] features;
        string[] answers;
        internal readonly int Width;
        internal readonly int Height;
        public DataReader()
        {
            string filePath = Path.Combine(currentDirectory, dataPath);
            if (!File.Exists(filePath))
            {
                allData.Add("Data not found.");
                return;
                
            }
            using (StreamReader reader = new StreamReader(filePath)) 
            {
                
                while (!reader.EndOfStream)
                {
                    string line = reader.ReadLine();
                    allData.Add(line);
                }
            }
            features = new double[allData.Count, names.Length - 1];
            answers = new string[allData.Count];
            int count = 0;
            foreach (string line in allData)
            {
                string[] splitLine = line.Split(",");
                for (int i = 0; i < splitLine.Length - 1; i++)
                {
                    features[count, i] = Double.Parse(splitLine[i]);
                }
                answers[count] = splitLine[^1];
                count++;
            }
            Width = names.Length;
            Height = allData.Count;
            return;
            
        }
        internal List<string> GetData()
        {
            return allData;
        }
        internal string[] GetNames()
        {
            return names;
        }
        internal string[] GetClassifications()
        {
            return classifications;
        }
        internal double[,] GetFeatures()
        {
            return features;
        }
        internal double[] GetRow(int ind)
        {
            int lengthRow = features.GetLength(1);
            double[] output = new double[lengthRow];
            for (int i = 0; i < lengthRow; i++)
            {
                output[i] = features[ind, i];
            }
            return output;
        }
        internal string[] GetAnswers()
        {
            return answers;
        }
    }
}
