using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Transforms;
using Microsoft.ML.Trainers;
using Common;

namespace edaAttrition
{
    class Program
    {

        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            IDataView attritionData = mlContext.Data.LoadFromTextFile<Employee>(path: "./data/attrition.csv", hasHeader: true, separatorChar: ',');

            var split = mlContext.Data.TrainTestSplit(attritionData, testFraction: 0.2);
            var trainData = split.TrainSet;
            var testData = split.TestSet;

            var modelPathName = "./model/attritionModel.zip";
            /*
            DataViewSchema modelSchema;
            ITransformer trainedModel;
            if (System.IO.File.Exists(modelPathName))
            {                
                trainedModel = mlContext.Model.Load(modelPathName, out modelSchema);
            }
            */

            var numFields = attritionData.Schema.AsEnumerable()
                .Select(column => new { column.Name, column.Type })
                .Where(column => (column.Name != nameof(Employee.Attrition)) && (column.Type.ToString() == "Single"))
                .ToArray();

            var numFieldNames = numFields.AsEnumerable()
                .Select(column => column.Name)
                .ToList();
            
            // var numFieldNames = new List<string>();
            numFieldNames.Add(nameof(Employee.BusinessTravel)+"-OHE");
            numFieldNames.Add(nameof(Employee.Department)+"-OHE");
            numFieldNames.Add(nameof(Employee.EducationField)+"-OHE");
            numFieldNames.Add(nameof(Employee.MaritalStatus)+"-OHE");
            numFieldNames.Add(nameof(Employee.JobLevel)+"-OHE");
            numFieldNames.Add(nameof(Employee.JobRole)+"-OHE");
            numFieldNames.Add(nameof(Employee.OverTime)+"-OHE");

            string[] numericFields = numFieldNames.ToArray();

            var dataPipeline = mlContext.Transforms.Categorical.OneHotEncoding(
                new[]
                {
                    new InputOutputColumnPair(nameof(Employee.BusinessTravel)+"-OHE", nameof(Employee.BusinessTravel)),
                    new InputOutputColumnPair(nameof(Employee.Department)+"-OHE", nameof(Employee.Department)),
                    new InputOutputColumnPair(nameof(Employee.EducationField)+"-OHE", nameof(Employee.EducationField)),
                    new InputOutputColumnPair(nameof(Employee.MaritalStatus)+"-OHE", nameof(Employee.MaritalStatus)),
                    new InputOutputColumnPair(nameof(Employee.JobLevel)+"-OHE", nameof(Employee.JobLevel)),
                    new InputOutputColumnPair(nameof(Employee.JobRole)+"-OHE", nameof(Employee.JobRole)),
                    new InputOutputColumnPair(nameof(Employee.OverTime)+"-OHE", nameof(Employee.OverTime))
                }, OneHotEncodingEstimator.OutputKind.Indicator)
                .Append(mlContext.Transforms.Concatenate("Features", numericFields));

                /*                    
                .Append(mlContext.Transforms.DropColumns(nameof(Employee.BusinessTravel), nameof(Employee.Department),
                                                        nameof(Employee.EducationField), nameof(Employee.MaritalStatus), 
                                                        nameof(Employee.JobLevel), nameof(Employee.JobRole), nameof(Employee.OverTime)));
                */

            ConsoleHelper.ConsoleWriteHeader("=============== Preparing Data ===============");
            var prepTrainData = dataPipeline.Fit(trainData).Transform(trainData);
            ConsoleHelper.ConsoleWriteHeader("=============== Prepared Data ===============");

            // var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
            //                    labelColumnName:nameof(Employee.Attrition), 
            //                    featureColumnName:"Features");

            var trainer = mlContext.BinaryClassification.Trainers.LightGbm(labelColumnName:nameof(Employee.Attrition));

            var trainPipeline = dataPipeline.Append(trainer);

            ConsoleHelper.ConsoleWriteHeader("=============== Training model ===============");
            var trainedModel = trainPipeline.Fit(trainData);
            mlContext.Model.Save(trainedModel, attritionData.Schema, modelPathName);

            ConsoleHelper.ConsoleWriteHeader("=============== Trained and Saved the model ===============");

            /*
            var viewTrainPipeline = mlContext.Transforms
                .CalculateFeatureContribution(model.LastTransformer)
                .Fit(dataPipeline.Fit(trainData).Transform(trainData));
            */

            Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
            var testDataPredictions = trainedModel.Transform(testData);

            var evaluateMetrics = mlContext.BinaryClassification.Evaluate(data: testDataPredictions, 
                                                                labelColumnName: nameof(Employee.Attrition), 
                                                                scoreColumnName: "Score");

            ConsoleHelper.PrintBinaryClassificationMetrics(trainedModel.ToString(), evaluateMetrics);            

            var permutationMetrics = mlContext.BinaryClassification.PermutationFeatureImportance(
                    predictionTransformer:trainedModel.LastTransformer,  
                    data:prepTrainData, 
                    labelColumnName:nameof(Employee.Attrition), 
                    permutationCount: 50);

            // Now let's look at which features are most important to the model
            // overall. Get the feature indices sorted by their impact on AUC.
            var sortedIndices = permutationMetrics
                .Select((metrics, index) => new { index, metrics.AreaUnderRocCurve})
                .OrderByDescending(
                feature => Math.Abs(feature.AreaUnderRocCurve.Mean))
                .Select(feature => feature.index);

            Console.WriteLine("Feature\tModel Weight\tChange in AUC"
                + "\t95% Confidence in the Mean Change in AUC");
            var auc = permutationMetrics.Select(x => x.AreaUnderRocCurve).ToArray();

            for(int i=0; i<10; i++)
            {
                Console.WriteLine("{0}\t{1:G4}\t{2:G4}",
                    numericFields[i],                    
                    auc[i].Mean,
                    1.96 * auc[i].StandardError);
            }
        }
    }
}
