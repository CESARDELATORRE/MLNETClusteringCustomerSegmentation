using System;
using CustomerSegmentation.Model;
using System.IO;
using System.Threading.Tasks;

namespace CustomerSegmentation
{
    public class Program
    {
        static async Task Main(string[] args)
        {
            // Running inside Visual Studio, $SolutionDir/assets is automatically passed as argument
            // If you execute from the console, pass as argument the location of the assets folder
            // Otherwise, it will search for assets in the executable's folder
            var assetsPath = args.Length > 0 ? args[0] : ModelHelpers.GetAssetsPath();

            var transactionsCsv = Path.Combine(assetsPath, "inputs", "transactions.csv");
            var offersCsv = Path.Combine(assetsPath, "inputs", "offers.csv");
            var modelZip = Path.Combine(assetsPath, "outputs", "retailClustering.zip");
            var plotSvg = Path.Combine(assetsPath, "outputs", "customerSegmentation.svg");

            try
            {
                var modelEvaluator = new ModelEvaluator(transactionsCsv, offersCsv, modelZip, plotSvg);
                await modelEvaluator.Evaluate();
            } catch (Exception ex)
            {
                Console.WriteLine($"Exception: {ex.Message}");
            }
            Console.ReadKey();
        }
    }
}
