using CustomerSegmentation.Model;
using Microsoft.ML.Runtime.Api;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace CustomerSegmentation.RetailData
{
    public class PivotData
    {
        [Column("0")]
        public float C1 { get; set; }
        [Column("1")]
        public float C2 { get; set; }
        [Column("2")]
        public float C3 { get; set; }
        [Column("3")]
        public float C4 { get; set; }
        [Column("4")]
        public float C5 { get; set; }
        [Column("5")]
        public float C6 { get; set; }
        [Column("6")]
        public float C7 { get; set; }
        [Column("7")]
        public float C8 { get; set; }
        [Column("8")]
        public float C9 { get; set; }
        [Column("9")]
        public float C10 { get; set; }
        [Column("10")]
        public float C11 { get; set; }
        [Column("11")]
        public float C12 { get; set; }
        [Column("12")]
        public float C13 { get; set; }
        [Column("13")]
        public float C14 { get; set; }
        [Column("14")]
        public float C15 { get; set; }
        [Column("15")]
        public float C16 { get; set; }
        [Column("16")]
        public float C17 { get; set; }
        [Column("17")]
        public float C18 { get; set; }
        [Column("18")]
        public float C19 { get; set; }
        [Column("19")]
        public float C20 { get; set; }
        [Column("20")]
        public float C21 { get; set; }
        [Column("21")]
        public float C22 { get; set; }
        [Column("22")]
        public float C23 { get; set; }
        [Column("23")]
        public float C24 { get; set; }
        [Column("24")]
        public float C25 { get; set; }
        [Column("25")]
        public float C26 { get; set; }
        [Column("26")]
        public float C27 { get; set; }
        [Column("27")]
        public float C28 { get; set; }
        [Column("28")]
        public float C29 { get; set; }
        [Column("29")]
        public float C30 { get; set; }
        [Column("30")]
        public float C31 { get; set; }
        [Column("31")]
        public float C32 { get; set; }
        [Column("32")]
        public string LastName { get; set; }

        public override string ToString()
        {
            return $"{C1},{C2},{C3},{C4},{C5},{C6},{C7},{C8},{C9}," +
                   $"{C10},{C11},{C12},{C13},{C14},{C15},{C16},{C17},{C18},{C19}," +
                   $"{C20},{C21},{C22},{C23},{C24},{C25},{C26},{C27},{C28},{C29}," +
                   $"{C31},{LastName}";
        }

        public static void SaveToCsv(IEnumerable<PivotData> salesData, string file)
        {
            var columns = "C1,C2,C3,C4,C5,C6,C8,C9," +
                          "C10,C11,C12,C13,C14,C15,C16,C17,C18,C19," +
                          "C20,C21,C22,C23,C24,C25,C26,C27,C28,C29," +
                          $"C30,C31,{nameof(LastName)}";

            File.WriteAllLines(file, salesData
                .Select(s => s.ToString())
                .Prepend(columns));
        }
    }
}
