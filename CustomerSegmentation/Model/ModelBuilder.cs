using CustomerSegmentation.RetailData;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Scikit.ML.DataFrame;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CustomerSegmentation.Model
{
    class ModelBuilder
    {
        private readonly string transactionsDataLocation;
        private readonly string offersDataLocation;

        public ModelBuilder(string transactionsDataLocation, string offersDataLocation)
        {
            this.transactionsDataLocation = transactionsDataLocation;
            this.offersDataLocation = offersDataLocation;
        }

        public void BuildAndTrain(int kClusters = 3)
        {
            var preProcessData = PreProcess();
            var learningPipeline = BuildModel(preProcessData, kClusters);
            var model = Train(learningPipeline);
        }

        protected PredictionModel<PivotData, ClusteringPrediction> Train(LearningPipeline pipeline)
        {
            var model = pipeline.Train<PivotData, ClusteringPrediction>();
            return model;
        }

        protected LearningPipeline BuildModel(IEnumerable<PivotData> pivotData, int kClusters)
        {
            var pipeline = new LearningPipeline();
            pipeline.Add(CollectionDataSource.Create(pivotData));

            var columnsNumerical = Helpers.ColumnsNumerical<PivotData>();
            pipeline.Add(new ColumnConcatenator(outputColumn: "NumericalFeatures", columnsNumerical));

            pipeline.Add(new PcaCalculator("NumericalFeatures", "PCAFeatures") { Rank = 2 });

            pipeline.Add(new ColumnConcatenator("Features", "NumericalFeatures", "PCAFeatures"));

            pipeline.Add(new KMeansPlusPlusClusterer() { K = kClusters });

            return pipeline;
        }

        protected IEnumerable<PivotData> PreProcess()
        {
            var offers = Offer.ReadFromCsv(offersDataLocation);
            var transactions = Transaction.ReadFromCsv(transactionsDataLocation);

            // join datasets
            var clusterData = (from of in offers
                               join tr in transactions on of.OfferId equals tr.OfferId
                               select new ClusterData()
                               {
                                   OfferId = of.OfferId,
                                   Campaign = of.Campaign,
                                   Discount = of.Discount,
                                   LastName = tr.LastName,
                                   LastPeak = of.LastPeak,
                                   Minimum = of.Minimum,
                                   Origin = of.Origin,
                                   Varietal = of.Varietal
                               }).ToArray();

            // pivot table
            var pivotData =
                (from c in clusterData
                 group c by c.LastName into gcs
                 let lookup = gcs.ToLookup(y => y.OfferId, y => y.Count)
                 select new PivotData()
                 {
                     //LastName = gcs.Key,
                     C1 = (float) lookup["1"].Sum(),
                     C2 = (float) lookup["2"].Sum(),
                     C3 = (float) lookup["3"].Sum(),
                     C4 = (float) lookup["4"].Sum(),
                     C5 = (float) lookup["5"].Sum(),
                     C6 = (float) lookup["6"].Sum(),
                     C7 = (float) lookup["7"].Sum(),
                     C8 = (float) lookup["8"].Sum(),
                     C9 = (float) lookup["9"].Sum(),
                     C10 = (float) lookup["10"].Sum(),
                     C11 = (float) lookup["11"].Sum(),
                     C12 = (float) lookup["12"].Sum(),
                     C13 = (float) lookup["13"].Sum(),
                     C14 = (float) lookup["14"].Sum(),
                     C15 = (float) lookup["15"].Sum(),
                     C16 = (float) lookup["16"].Sum(),
                     C17 = (float) lookup["17"].Sum(),
                     C18 = (float) lookup["18"].Sum(),
                     C19 = (float) lookup["19"].Sum(),
                     C20 = (float) lookup["20"].Sum(),
                     C21 = (float) lookup["21"].Sum(),
                     C22 = (float) lookup["22"].Sum(),
                     C23 = (float) lookup["23"].Sum(),
                     C24 = (float) lookup["24"].Sum(),
                     C25 = (float) lookup["25"].Sum(),
                     C26 = (float) lookup["26"].Sum(),
                     C27 = (float) lookup["27"].Sum(),
                     C28 = (float) lookup["28"].Sum(),
                     C29 = (float) lookup["29"].Sum(),
                     C30 = (float) lookup["30"].Sum(),
                     C31 = (float) lookup["31"].Sum(),
                     C32 = (float) lookup["32"].Sum()
                 }).ToArray();

            return pivotData;
        }
    }
}
