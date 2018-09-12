using CustomerSegmentation.RetailData;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using OxyPlot;
using OxyPlot.Series;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CustomerSegmentation.Model
{
    public class ModelEvaluator
    {
        private readonly string transactionsDataLocation;
        private readonly string offersDataLocation;
        private readonly string modelLocation;
        private readonly string plotLocation;

        public ModelEvaluator(string transactionsDataLocation, string offersDataLocation, string modelLocation, string plotLocation)
        {
            this.transactionsDataLocation = transactionsDataLocation;
            this.offersDataLocation = offersDataLocation;
            this.modelLocation = modelLocation;
            this.plotLocation = plotLocation;
        }

        public async Task Evaluate()
        {
            var preProcessData = DataHelpers.PreProcess(offersDataLocation, transactionsDataLocation);
            var model = await PredictionModel.ReadAsync<PivotData, ClusteringPrediction>(modelLocation);
            var predictions = GetPredictions(preProcessData, model).ToArray();
            ShowMetrics(preProcessData, model);
            WritePlot(predictions, plotLocation);
        }

        public void ShowMetrics(IEnumerable<PivotData> testData, PredictionModel<PivotData, ClusteringPrediction> model)
        {
            var evaluator = new ClusterEvaluator();
            var testDataSource = CollectionDataSource.Create(testData);
            ClusterMetrics metrics = evaluator.Evaluate(model, testDataSource);
            PrintMetrics(metrics);
        }

        public IEnumerable<ClusteringPrediction> GetPredictions(IEnumerable<PivotData> testData, PredictionModel<PivotData, ClusteringPrediction> model)
        {
            foreach (var sample in testData)
            {
                yield return model.Predict(sample);
            }
        }

        private static void PrintMetrics(ClusterMetrics metrics)
        {
            Console.WriteLine($"**************************************************************");
            Console.WriteLine($"*       Metrics for Retail Clustering          ");
            Console.WriteLine($"*-------------------------------------------------------------");
            Console.WriteLine($"*       Average mean score: {metrics.AvgMinScore:0.##}");
            Console.WriteLine($"*       Davies-Bouldin Index: {metrics.Dbi:#.##}");
            Console.WriteLine($"*       Normalized mutual information: {metrics.Nmi:#.##}");
            Console.WriteLine($"**************************************************************");
        }

        private static void WritePlot(IEnumerable<ClusteringPrediction> predictions, string plotLocation)
        {
            var plot = new PlotModel { Title = "Customer Segmentation" };

            var clusters = predictions.Select(p => p.SelectedClusterId).Distinct();

            foreach (var item in new[] { (ClusterId: 1, MarkerType: MarkerType.Circle),
                                         (ClusterId: 2, MarkerType: MarkerType.Diamond),
                                         (ClusterId: 3, MarkerType: MarkerType.Square),
                                         (ClusterId: 4, MarkerType: MarkerType.Triangle),
                                         (ClusterId: 5, MarkerType: MarkerType.Star) })
            {
                var scatter = new ScatterSeries { MarkerType = item.MarkerType, MarkerStrokeThickness = 2 };
                var series = predictions
                    .Where(p => p.SelectedClusterId == item.ClusterId)
                    .Select(p => new ScatterPoint(p.PCAFeatures[0], p.PCAFeatures[1])).ToArray();
                scatter.Points.AddRange(series);
                plot.Series.Add(scatter);
            }

            var exporter = new SvgExporter { Width = 600, Height = 400 };
            using (var fs = new System.IO.FileStream(plotLocation, System.IO.FileMode.Create))
            {
                exporter.Export(plot, fs);
            }
        }
    }
}
