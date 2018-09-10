using CustomerSegmentation.Model;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace CustomerSegmentation
{
    class Program
    {
        static void Main(string[] args)
        {
            //ClusteringTraining();
            var modelBuilder = new ModelBuilder(Helpers.GetAssetsPath("data", "transactions.csv"), Helpers.GetAssetsPath("data","offers.csv"));
            modelBuilder.BuildAndTrain();
        }
    }
}
