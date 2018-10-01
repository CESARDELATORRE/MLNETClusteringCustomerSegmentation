using System;
using CustomerSegmentation.RetailData;
using System.Collections.Generic;
using System.Threading.Tasks;
using static CustomerSegmentation.Model.ConsoleHelpers;
using OxyPlot;
using OxyPlot.Series;
using System.Linq;
using OxyPlot.Axes;
using Microsoft.ML.Runtime.KMeans;
using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.StaticPipe;
using System.IO;
using Microsoft.ML.Core.Data;

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
                                Features: c.LoadFloat(0, 29),
                                LastName: c.LoadText(30)),
                            separator: ',', hasHeader: true);

            KMeansPredictor pred = null;
            var clustering = new ClusteringContext(env);

            var est = reader.MakeNewEstimator()
                .Append(row => (row.LastName, LastNameKey: row.LastName.ToKey(), row.Features, PCAFeatures: row.Features.ToPrincipalComponents(rank: 2, (p) => p.Seed = 42)))
                .Append(row => (row.LastName, row.LastNameKey, row.PCAFeatures, row.Features,
                preds: clustering.Trainers.KMeans(
                    row.Features, clustersCount: kClusters,
                    onFit: p => pred = p)));

            ConsoleWriteHeader("Training model for recommendations");
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


            ITransformer testPredictor;
            using (var file = File.OpenRead(modelLocation))
            {
                testPredictor = TransformerChain
                    .LoadFrom(env, file);
            }
        }

        //protected void PlotKValues(Dictionary<int,double> kValues, string plotLocation)
        //{
        //    ConsoleWriteHeader("Plot Customer Segmentation");

        //    var plot = new PlotModel { Title = "elbow method", IsLegendVisible = true };

        //    var lineSeries = new LineSeries() { Title = $"kValues ({kValues.Keys.Max()})" };
        //    foreach (var item in kValues)
        //        lineSeries.Points.Add(new DataPoint(item.Key, item.Value));

        //    plot.Series.Add(lineSeries);
        //    plot.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Minimum = -0.1, Title = "k" });
        //    plot.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Minimum = -0.1, Title = "loss" });

        //    var exporter = new SvgExporter { Width = 600, Height = 400 };
        //    using (var fs = new System.IO.FileStream(plotLocation, System.IO.FileMode.Create))
        //    {
        //        exporter.Export(plot, fs);
        //    }

        //    Console.WriteLine($"Plot location: {plotLocation}");
        //}

    }
}
