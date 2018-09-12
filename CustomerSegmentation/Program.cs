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
using System.Threading.Tasks;

namespace CustomerSegmentation
{
    class Program
    {
        static async Task Main(string[] args)
        {
            var transactionsCsv = ModelHelpers.GetAssetsPath("data", "transactions.csv");
            var offersCsv = ModelHelpers.GetAssetsPath("data", "offers.csv");
            var modelZip = ModelHelpers.GetAssetsPath("model", "retailClustering.zip");
            var plotSvg = ModelHelpers.GetAssetsPath("model", "series.svg");

            var modelBuilder = new ModelBuilder(transactionsCsv, offersCsv, modelZip);
            await modelBuilder.BuildAndTrain();

            var modelEvaluator = new ModelEvaluator(transactionsCsv, offersCsv, modelZip, plotSvg);
            await modelEvaluator.Evaluate();
        }
    }
}
