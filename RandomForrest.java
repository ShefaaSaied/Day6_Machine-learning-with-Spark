package jupai9.examples;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.sql.DataFrameReader;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class RandomForrest {
    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.ERROR);
        // Create Spark Session to create connection to Spark
        final SparkSession sparkSession = SparkSession.builder().appName("AirBnb DT").master("local[6]").getOrCreate();
        // Get DataFrameReader using SparkSession
        final DataFrameReader dataFrameReader = sparkSession.read();
        // Set header option to true to specify that first row in file contains
        // name of columns
        dataFrameReader.option("header", "true");
        Dataset<Row> airbnbDF = dataFrameReader.csv("src/main/resources/listings.csv");
        //============================================================================================================
        airbnbDF = airbnbDF.select("id", "neighbourhood", "room_type", "bedrooms", "minimum_nights",
                "number_of_reviews", "price");
        //============================================================================================================
        // Split the data into training and test sets
        double split[] = {0.8, 0.2};
        Dataset<Row>[] splits = airbnbDF.randomSplit(split, 42);
        Dataset<Row> airbnbDFTrain = splits[0];
        Dataset<Row> airbnbDFTest = splits[1];

        //ensure that the Train data set field bedrooms  is double and that it does not contain nulls
        airbnbDFTrain = airbnbDFTrain.withColumn("id", airbnbDFTrain.col("id").cast("double"))
                .filter(airbnbDFTrain.col("id").isNotNull());
        airbnbDFTrain = airbnbDFTrain.withColumn("minimum_nights", airbnbDFTrain.col("minimum_nights").cast("double"))
                .filter(airbnbDFTrain.col("minimum_nights").isNotNull());
        airbnbDFTrain = airbnbDFTrain.withColumn("number_of_reviews", airbnbDFTrain.col("number_of_reviews").cast("double"))
                .filter(airbnbDFTrain.col("number_of_reviews").isNotNull());
        airbnbDFTrain = airbnbDFTrain.withColumn("bedrooms", airbnbDFTrain.col("bedrooms").cast("double"))
                .filter(airbnbDFTrain.col("bedrooms").isNotNull());
        airbnbDFTrain = airbnbDFTrain.withColumn("price", airbnbDFTrain.col("price").cast("double"))
                .filter(airbnbDFTrain.col("price").isNotNull());
        airbnbDFTrain.printSchema();
        //============================================================================================================
        //Getting the Test Dataset
        //ensure that the Test data set field bedrooms  is double and that it does not contain nulls
        airbnbDFTest = airbnbDFTest.withColumn("id", airbnbDFTest.col("id").cast("double"))
                .filter(airbnbDFTest.col("id").isNotNull());
        airbnbDFTest = airbnbDFTest.withColumn("minimum_nights", airbnbDFTest.col("minimum_nights").cast("double"))
                .filter(airbnbDFTest.col("minimum_nights").isNotNull());
        airbnbDFTest = airbnbDFTest.withColumn("number_of_reviews", airbnbDFTest.col("number_of_reviews").cast("double"))
                .filter(airbnbDFTest.col("number_of_reviews").isNotNull());
        airbnbDFTest = airbnbDFTest.withColumn("bedrooms", airbnbDFTest.col("bedrooms").cast("double"))
                .filter(airbnbDFTest.col("bedrooms").isNotNull());
        airbnbDFTest = airbnbDFTest.withColumn("price", airbnbDFTest.col("price").cast("double"))
                .filter(airbnbDFTest.col("price").isNotNull());
        airbnbDFTest.printSchema();
        //============================================================================================================
        //Create the Vector Assembler That will contain the feature columns
        String inputColumns[] = {"bedrooms"};
        VectorAssembler vectorAssembler = new VectorAssembler();
        vectorAssembler.setInputCols(inputColumns);
        vectorAssembler.setOutputCol("features");
        //============================================================================================================
        //Transform the Train Dataset using vectorAssembler.transform
        Dataset<Row> airbnbDFTrainTransform = vectorAssembler.transform(airbnbDFTrain.na().drop());
        airbnbDFTrainTransform.select("bedrooms", "features", "price").show(10);
        //============================================================================================================
        ////////////////////////////////////////////// Random Forest /////////////////////////////////////////////////
        // Automatically identify categorical features, and index them.
        // Set maxCategories so features with > 4 distinct values are treated as continuous.
        VectorIndexerModel featureIndexer = new VectorIndexer()
                .setInputCol("features")
                .setOutputCol("indexedFeatures")
                .setMaxCategories(4)
                .fit(airbnbDFTrainTransform);

        // Train a RandomForest model.
        RandomForestRegressor rf = new RandomForestRegressor();
        rf.setFeaturesCol ("features");
        rf.setLabelCol ("price");

        // Chain indexer and forest in a Pipeline
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {featureIndexer, rf});

        // Train model. This also runs the indexer.
        PipelineModel model = pipeline.fit(airbnbDFTrainTransform);

        // Make predictions.
        airbnbDFTest=vectorAssembler.transform (airbnbDFTest.na().drop());
        Dataset<Row> predictions = model.transform(airbnbDFTest);

        predictions.show();

        // Select (prediction, true label) and compute test error
        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("price")
                .setPredictionCol("prediction")
                .setMetricName("rmse");
        double rmse = evaluator.evaluate(predictions);
        System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmse);
        RandomForestRegressionModel rfModel = (RandomForestRegressionModel)(model.stages()[1]);
        System.out.println("Learned regression forest model:\n" + rfModel.toDebugString());
    }
}
