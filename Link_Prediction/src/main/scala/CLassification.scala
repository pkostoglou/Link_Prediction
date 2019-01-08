import org.apache.log4j.{Level, Logger}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer,Normalizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{split,udf,concat_ws}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.feature.MinMaxScaler


// For implicit conversions like converting RDDs to DataFrames


object Classification {

  def sqrdist(vec1: Vector , vec2: Vector):Double={
    val temp1 = vec1.toArray
    val temp2 = vec2.toArray
    return temp1.zip(temp2).map(x=>x._1*x._2).reduce((a,b)=>a+b)
  }

  def sameAuthor(auth1: String,auth2: String):Int={
    if(auth1==null || auth2 ==null)
      return 0
    val temp1=auth1.split(",")
    val temp2=auth2.split(",")
    for( i <- 0 until temp1.length){
      for(j <-0 until temp2.length){
        if(temp1(i).compareTo(temp2(j))==0){
          return 1
        }
      }
    }
    return 0
  }

  def sameJournal(Journal1: String,Journal2: String):Int={
    if(Journal1==null || Journal2 ==null)
      return 0

    if(Journal1.compareTo(Journal2)==0){
      return 1
    }

    return 0
  }


  //get a dataframe and use tf idf at a specific column
  def TFIDF(x: DataFrame,column: String,columnOut: String,numFeatures: Int):DataFrame = {
    val tokenizer = new Tokenizer().setInputCol(column).setOutputCol("words")
    val wordsData = tokenizer.transform(x.na.fill(Map(column -> "")))

    val hashingTF = new HashingTF()
      .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(numFeatures)

    val featurizedData = hashingTF.transform(wordsData)

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol(columnOut)
    val idfModel = idf.fit(featurizedData)

    val rescaledData = idfModel.transform(featurizedData)
    return rescaledData.drop("words").drop("rawFeatures").drop(column)
  }


  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new spark.SparkConf().setAppName("Link Classification Application").setMaster("local[*]")
    val sc = new spark.SparkContext(conf)
    val sparkSession = SparkSession
      .builder()
      .appName("Spark SQL basic example")
      .config("spark.some.config.option", "some-value")
      .enableHiveSupport()
      .getOrCreate()

    import sparkSession.implicits._
    import sparkSession.sql

    //load node information
    val dfHeaderless = sparkSession.read.format("csv").option("header","false").load("Data-20181224/node_information.csv")
    val dfNodeInformation = dfHeaderless.toDF("Id","Year","Title","Authors","Journal","Abstract")
    //load training set
    val df2 = sparkSession.read.format("csv").option("header","false").load("Data-20181224/training_set.txt").toDF("info")

    def ProcessDf(NodeInfo: DataFrame,NodeCombination: DataFrame):DataFrame = {
      //break NodeCombination's info into 3 columns(ID1,Id2,label)
      val dfComSplit=NodeCombination.withColumn("_tmp", split($"info", " ")).select(
        $"_tmp".getItem(0).as("Id1"),
        $"_tmp".getItem(1).as("Id2"),
        $"_tmp".getItem(2).as("label")
      ).drop("_tmp")

      //join the information for every node with id=Id1 rename column names and change the type of column label from string to Int
      val test=dfComSplit.join(NodeInfo,$"Id1"===$"Id").select($"Id1",$"Id2",$"label",$"fAbstract".as("fAbstract1"),
        $"Authors".as("Authors1"),$"fTitle".as("Title1"),$"Journal".as("Journal1"))
        .withColumn("label",dfComSplit.col("label").cast(IntegerType))
      //join the information for every node with id=Id2 rename column names
      val lastDf=test.join(NodeInfo,$"Id2"===$"Id").select($"Id1",$"Id2",$"label",$"fAbstract1", $"Authors1",$"Title1",$"Journal1",
        $"fAbstract".as("fAbstract2"),$"Authors".as("Authors2"),$"fTitle".as("Title2"),$"Journal".as("Journal2"))
      val dist = udf(sqrdist _ )
      val sAuthor= udf(sameAuthor _)
      val sJournal = udf(sameJournal _ )
      //produce final dataframe calculate euclidean distance between abstracts and check if there is the same author between 2 articles
      val finalDf=lastDf.withColumn("Dist",dist($"fAbstract1",$"fAbstract2"))
        .withColumn("DistTitle",dist($"Title1",$"Title2"))
        .withColumn("sameAuthors",sAuthor($"Authors1",$"Authors2"))
        .withColumn("Id",concat_ws(" ",$"Id1",$"Id2"))
        .withColumn("sameJournal", sJournal($"Journal1",$"Journal2"))
        .select("Id","label","Dist","DistTitle","sameAuthors","sameJournal")
      //dataframe's last form is Id,label,Euclidean distance,same authors
      return finalDf

    }

    //process for the abstact
    val dfNodeInfoProstemp=TFIDF(dfNodeInformation,"Abstract","fAbstract",1000).drop("Abstract")
    val dfNodeInfoPros=TFIDF(dfNodeInfoProstemp,"Title","fTitle",1000).drop("Title")
    //process the dataframe
    val ProcessedDf=ProcessDf(dfNodeInfoPros,df2)

    /*val normalizer = new Normalizer()
      .setInputCol("featurestemp")
      .setOutputCol("features")
      .setP(1.0)*/

    val assembler1 = new VectorAssembler().
      setInputCols(Array("Dist","DistTitle","sameAuthors","sameJournal")).
      setOutputCol("features")


    val assembledFinalDf = assembler1.transform(ProcessedDf).select("Id","features","label")

    //split the training for train and test
    val Array(training, test) = assembledFinalDf.randomSplit(Array[Double](0.7, 0.3), 18)

    //decision tree model
    val dt = new DecisionTreeClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
    val trained=dt.fit(training)
    val prediction=trained.transform(test)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("f1")
    val accuracy = evaluator.evaluate(prediction)
    println(s"F1 score = ${accuracy}")


    /*val lsvc = new LinearSVC()
      .setMaxIter(10)
      .setRegParam(0.1)

    // Fit the model
    val lsvcModel = lsvc.fit(training)

    // Print the coefficients and intercept for linear svc
    println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")

    val prediction=lsvcModel.transform(test)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(prediction)
    println(s"Test acc = ${accuracy}")*/


  }
}

