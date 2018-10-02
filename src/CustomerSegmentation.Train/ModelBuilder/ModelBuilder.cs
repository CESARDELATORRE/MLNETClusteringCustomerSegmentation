using System;
using static CustomerSegmentation.Model.ConsoleHelpers;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.StaticPipe;
using System.IO;

namespace CustomerSegmentation.Model
{
    public class ModelBuilder
    {
        private readonly string pivotLocation;
        private readonly string modelLocation;
        private readonly string plotLocation;
        private readonly LocalEnvironment env;

        public ModelBuilder(string pivotLocation, string modelLocation, string plotLocation)
        {
            this.pivotLocation = pivotLocation;
            this.modelLocation = modelLocation;
            this.plotLocation = plotLocation;
            env = new LocalEnvironment(seed: 1);  //Seed set to any number so you have a deterministic environment
        }

        private class PivotPipelineData {
            // Features,LastName,PCAFeatures,PredictedLabel,Score,preds.score,preds.predictedLabel
            public float[] Features;
            public string LastName;
            public float[] PCAFeatures;
            public float[] Score;
        }

        public void BuildAndTrain(int kClusters = 4)
        {
            ConsoleWriteHeader("Build and Train using Static API");
            Console.Out.WriteLine($"Input file: {pivotLocation}");

            ConsoleWriteHeader("Reading file ...");
            var reader = TextLoader.CreateReader(env,
                            c => (
                                Features: c.LoadFloat(0, 31),
                                LastName: c.LoadText(32)),
                            separator: ',', hasHeader: true);

            var clustering = new ClusteringContext(env);

            var est = reader.MakeNewEstimator()
                .Append(row => (row.LastName, 
                                LastNameKey: row.LastName.ToKey(), 
                                row.Features, 
                                PCAFeatures: row.Features.ToPrincipalComponents(rank: 2, (p) => p.Seed = 42)))
                .Append(row => (row.LastName, 
                                row.LastNameKey, 
                                row.PCAFeatures, 
                                row.Features,
                                preds: clustering.Trainers.KMeans(row.Features, clustersCount: kClusters)));

            ConsoleWriteHeader("Training model for customer clustering");
            var dataSource = reader.Read(new MultiFileSource(pivotLocation));

            var model = est.Fit(dataSource);

            // inspect data
            var data = model.Transform(dataSource);
            var trainData = data.AsDynamic;
            var columnNames = trainData.Schema.GetColumnNames().ToArray();
            var trainDataAsEnumerable = trainData.AsEnumerable<PivotPipelineData>(env, false).Take(10).ToArray();

            ConsoleWriteHeader("Evaluate model");
            var metrics = clustering.Evaluate(data, r => r.preds.score, r => r.LastNameKey, r => r.Features);
            Console.WriteLine($"AvgMinScore is: {metrics.AvgMinScore}");
            Console.WriteLine($"Dbi is: {metrics.Dbi}");

            ConsoleWriteHeader("Save model to local file");
            ModelHelpers.DeleteAssets(modelLocation);
            using (var f = new FileStream(modelLocation, FileMode.Create))
                model.AsDynamic.SaveTo(env, f);
            Console.WriteLine($"Model saved: {modelLocation}");
        }
    }
}
