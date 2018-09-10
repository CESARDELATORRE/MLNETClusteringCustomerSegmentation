using Microsoft.ML.Runtime.Api;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace CustomerSegmentation.RetailData
{
    public class Transaction
    {
        //Customer Last Name,Offer #
        //Smith,2
        [Column("0")]
        public string LastName { get; set; }
        [Column("1")]
        public string OfferId { get; set; }

        public static IEnumerable<Transaction> ReadFromCsv(string file)
        {
            return File.ReadAllLines(file)
             .Skip(1) // skip header
             .Select(x => x.Split(','))
             .Select(x => new Transaction()
             {
                 LastName = x[0],
                 OfferId = x[1],
             });
        }
    }

    public class ClusterData : Offer
    {
        [Column("7")]
        public string LastName { get; set; }

        public int Count { get; set; } = 1;
    }
}
