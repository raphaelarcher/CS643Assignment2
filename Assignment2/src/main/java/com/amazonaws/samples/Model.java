package com.amazonaws.samples;

import java.io.IOException;
import java.util.Scanner;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.DataFrameReader;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class Model {
	public static void main(String[] args) throws IOException {
		
		// Doesn't display logs up to "WARN" level
		Logger.getLogger("org.apache").setLevel(Level.WARN);
		// Create spark session with configurations
		SparkSession ss = SparkSession.builder().appName("Assignment-2").master("local[*]").getOrCreate();
		// Create DataFrameReader
		DataFrameReader dataFrameReader = ss.read();
		// When reading .csv file first row is a header
		// Let DataFrameReader determine the datatype for each attribute
		dataFrameReader.option("header", "true").option("sep", ";").option("inferSchema", "true");
		// Read dataset from TrainingDataset.csv file
		Dataset<Row> trainingData = dataFrameReader.csv("TrainingDataset.csv");
		// Show first 20 records
		trainingData.show();
		
		// Takes double values in columns and puts them into a single vector column
		// This column is used to train the model
		VectorAssembler assembler = new VectorAssembler()
				.setInputCols(new String[]{"\"\"\"\"\"fixed acidity\"\"\"\"", "\"\"\"\"volatile acidity\"\"\"\"", "\"\"\"\"citric acid\"\"\"\"",
						"\"\"\"\"residual sugar\"\"\"\"", "\"\"\"\"chlorides\"\"\"\"", "\"\"\"\"free sulfur dioxide\"\"\"\"", "\"\"\"\"total sulfur dioxide\"\"\"\"",
						"\"\"\"\"density\"\"\"\"", "\"\"\"\"pH\"\"\"\"", "\"\"\"\"sulphates\"\"\"\"", "\"\"\"\"alcohol\"\"\"\""})
				.setOutputCol("features");
		
		// Set up Random forest classifier algorithm to predict "quality" attribute
		RandomForestClassifier rf = new RandomForestClassifier()
				.setLabelCol("\"\"\"\"quality\"\"\"\"\"")
				.setFeaturesCol("features");
		
		// Puts VectorAssembler and Random forest in a pipeline
		Pipeline pipelineFull = new Pipeline().setStages(new PipelineStage[] {assembler, rf});
		
		// Trains model using "TrainingDataset.csv"
		PipelineModel model = pipelineFull.fit(trainingData);
		
		// Creates Scanner object to get user input for .csv testing file
		Scanner scanner = new Scanner(System.in);
	    System.out.print("Enter path to the test file: ");
	    String input = scanner.nextLine();
		
	    // Reads dataset from user input file
		Dataset<Row> testingData = dataFrameReader.csv(input);
		
		// Creates predictions
		Dataset<Row> predictions = model.transform(testingData);
		// Show 50 predictions compared to actual result
		predictions.select("\"\"\"\"quality\"\"\"\"\"", "Prediction").show(50);
			
		// Evaluates algorithm compared to actual results
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
				.setLabelCol("\"\"\"\"quality\"\"\"\"\"")
				.setPredictionCol("prediction")
				.setMetricName("accuracy");
			
		System.out.println("Accuracy: " + evaluator.evaluate(predictions));
		
		scanner.close();
		
		// Save model in a directory
		model.save("SavedModel");
  }
}