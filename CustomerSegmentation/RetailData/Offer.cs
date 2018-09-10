﻿using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace CustomerSegmentation.RetailData
{
    public class Offer
    {
        //Offer #,Campaign,Varietal,Minimum Qty (kg),Discount (%),Origin,Past Peak
        [Column("0")]
        public string OfferId { get; set; }
        [Column("1")]
        public string Campaign { get; set; }
        [Column("2")]
        public string Varietal { get; set; }
        [Column("3")]
        public float Minimum { get; set; }
        [Column("4")]
        public float Discount { get; set; }
        [Column("5")]
        public string Origin { get; set; }
        [Column("6")]
        public string LastPeak { get; set; }

        public static IEnumerable<Offer> ReadFromCsv(string file)
        {
            return File.ReadAllLines(file)
             .Skip(1) // skip header
             .Select(x => x.Split(','))
             .Select(x => new Offer()
             {
                 OfferId = x[0],
                 Campaign = x[1],
                 Varietal = x[2],
                 Minimum = float.Parse(x[3]),
                 Discount = float.Parse(x[4]),
                 Origin = x[5],
                 LastPeak = x[6]
             });
        }
    }
}
