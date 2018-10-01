using CustomerSegmentation.RetailData;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

namespace CustomerSegmentation.Model
{
    public static class ModelHelpers
    {
        static FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);

        public static string GetAssetsPath(params string[] paths)
        {
            if (paths == null || paths.Length == 0)
                return null;

            return Path.Combine(paths.Prepend(_dataRoot.Directory.FullName).ToArray());
        }

        public static string DeleteAssets(params string[] paths)
        {
            var location = GetAssetsPath(paths);

            if (!string.IsNullOrWhiteSpace(location) && File.Exists(location))
                File.Delete(location);
            return location;
        }

        public static IEnumerable<string> Columns<T>() where T : class
        {
            return typeof(T).GetProperties().Select(p => p.Name);
        }

        public static IEnumerable<string> Columns<T, U>() where T : class
        {
            var typeofU = typeof(U);
            return typeof(T).GetProperties().Where(c => c.PropertyType == typeofU).Select(p => p.Name);
        }

        public static IEnumerable<string> Columns<T, U, V>() where T : class
        {
            var typeofUV = new[] { typeof(U), typeof(V) };
            return typeof(T).GetProperties().Where(c => typeofUV.Contains(c.PropertyType)).Select(p => p.Name);
        }

        public static IEnumerable<string> Columns<T, U, V, W>() where T : class
        {
            var typeofUVW = new[] { typeof(U), typeof(V), typeof(W) };
            return typeof(T).GetProperties().Where(c => typeofUVW.Contains(c.PropertyType)).Select(p => p.Name);
        }

        public static string[] ColumnsNumerical<T>() where T : class
        {
            return Columns<T, float, int>().ToArray();
        }

        public static string[] ColumnsString<T>() where T : class
        {
            return Columns<T, string>().ToArray();
        }

        public static string[] ColumnsDateTime<T>() where T : class
        {
            return Columns<T, DateTime>().ToArray();
        }

        public static IEnumerable<string> GetColumnNames(this ISchema schema)
        {
            for (int i = 0; i < schema.ColumnCount; i++)
            {
                if (!schema.IsHidden(i))
                    yield return schema.GetColumnName(i);
            }
        }
    }

    public static class ConsoleHelpers
    {
        public static void ConsoleWriteHeader(params string[] lines)
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine(" ");
            foreach (var line in lines)
            {
                Console.WriteLine(line);
            }
            var maxLength = lines.Select(x => x.Length).Max();
            Console.WriteLine(new String('#', maxLength));
            Console.ForegroundColor = defaultColor;
        }

        public static void ConsolePressAnyKey()
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine(" ");
            Console.WriteLine("Press any key to finish.");
            Console.ReadKey();
        }

        public static void ConsoleWriteException(params string[] lines)
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Red;
            const string exceptionTitle = "EXCEPTION";
            Console.WriteLine(" ");
            Console.WriteLine(exceptionTitle);
            Console.WriteLine(new String('#', exceptionTitle.Length));
            Console.ForegroundColor = defaultColor;
            foreach (var line in lines)
            {
                Console.WriteLine(line);
            }
        }
    }

}
