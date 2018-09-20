using CustomerSegmentation.RetailData;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using OxyPlot;
using OxyPlot.Series;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using static CustomerSegmentation.Model.ModelHelpers;

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

            PredictionModel<PivotData, ClusteringPrediction> model = await LoadModel();

            IEnumerable<ClusteringPrediction> predictions = PredictDataUsingModel(preProcessData, model);

            EvaluateModel(preProcessData, model);

            SaveCustomerSegmentationPlot(predictions, plotLocation);
        }

        private static IEnumerable<ClusteringPrediction> PredictDataUsingModel(IEnumerable<PivotData> preProcessData, PredictionModel<PivotData, ClusteringPrediction> model)
        {
            ConsoleWriteHeader("Calculate customer segmentation");
            var predictions = model.Predict(preProcessData);
            return predictions;
        }

        private async Task<PredictionModel<PivotData, ClusteringPrediction>> LoadModel()
        {
            ConsoleWriteHeader("Read model");
            Console.WriteLine($"Model location: {modelLocation}");
            return await PredictionModel.ReadAsync<PivotData, ClusteringPrediction>(modelLocation);
        }

        public void EvaluateModel(IEnumerable<PivotData> testData, PredictionModel<PivotData, ClusteringPrediction> model)
        {
            ConsoleWriteHeader("Metrics for Customer Segmentation");
            var testDataSource = CollectionDataSource.Create(testData);
            var evaluator = new ClusterEvaluator { CalculateDbi = true };
            ClusterMetrics metrics = evaluator.Evaluate(model, testDataSource);
            Console.WriteLine($"Average mean score: {metrics.AvgMinScore:0.##}");
            Console.WriteLine($"Davies-Bouldin Index: {metrics.Dbi:0.##}");
        }

        private static void SaveCustomerSegmentationPlot(IEnumerable<ClusteringPrediction> predictions, string plotLocation)
        {
            ConsoleWriteHeader("Plot Customer Segmentation");

            var plot = new PlotModel { Title = "Customer Segmentation", IsLegendVisible = true };

            var clusters = predictions.Select(p => p.SelectedClusterId).Distinct().OrderBy(x => x);

            foreach (var cluster in clusters)
            {
                var scatter = new ScatterSeries { MarkerType = MarkerType.Circle, MarkerStrokeThickness = 2, Title = $"Cluster: {cluster}", RenderInLegend=true };
                var series = predictions
                    .Where(p => p.SelectedClusterId == cluster)
                    .Select(p => new ScatterPoint(p.Location[0], p.Location[1])).ToArray();
                scatter.Points.AddRange(series);
                plot.Series.Add(scatter);
            }

            plot.DefaultColors = OxyPalettes.HueDistinct(plot.Series.Count).Colors;

            var exporter = new SvgExporter { Width = 600, Height = 400 };
            using (var fs = new System.IO.FileStream(plotLocation, System.IO.FileMode.Create))
            {
                exporter.Export(plot, fs);
            }

            Console.WriteLine($"Plot location: {plotLocation}");
        }
    }
}
