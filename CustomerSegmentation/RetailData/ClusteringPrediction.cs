﻿using Microsoft.ML.Runtime.Api;

namespace CustomerSegmentation.RetailData
{
    public class ClusteringPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint SelectedClusterId;
        [ColumnName("Score")]
        public float[] Distance;
    }

}
