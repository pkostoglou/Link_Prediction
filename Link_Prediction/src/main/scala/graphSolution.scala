import org.apache.log4j.{Level, Logger}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.sql.types._
import org.apache.spark.ml.classification.{DecisionTreeClassifier, LinearSVC, LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{concat_ws, split, udf}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator

import scala.collection.mutable.WrappedArray

object graphSolution {

  def comparison(cluster1: Int, cluster2: Int):Int={
    if(cluster1==cluster2){
      return 1
    }else{
      return 0
    }

  }

  def inLinksDifference(links1:Int, links2:Int):Int={
    return math.abs(links1-links2)
  }

  def Neighbours(N1: WrappedArray[Long], N2: WrappedArray[Long]):Int={
    return N1.intersect(N2).length
  }

  def sqrdist(vec1: Vector , vec2: Vector):Double={
    val temp1 = vec1.toArray
    val temp2 = vec2.toArray
    return math.sqrt(temp1.zip(temp2).map(x=>math.pow(x._1-x._2,2)).reduce((a,b)=>a+b))
  }

  def cosdist(vec1: Vector , vec2: Vector):Double= {
    val temp1 = vec1.toArray
    val temp2 = vec2.toArray
    val res = temp1.zip(temp2).map(x=>(x._1*x._2,x._1*x._1,x._2*x._2))reduceLeft((a,b)=>(a._1+b._1,a._2+b._2,a._3+b._3))
    return res._1/(res._2*res._3)
  }


  def jaccard(vec1: Vector , vec2: Vector):Double= {
    val temp1 = vec1.toArray
    val temp2 = vec2.toArray
    return 1-temp1.intersect(temp2).size/temp1.union(temp2).size
  }

  def sameAuthor(auth1: String,auth2: String):Int={
    if(auth1==null || auth2 ==null)
      return 0
    val temp1=auth1.split(",")
    val temp2=auth2.split(",")
    var sameAuthors=0
    for( i <- 0 until temp1.length){
      for(j <-0 until temp2.length){
        if(temp1(i).compareTo(temp2(j))==0){
          sameAuthors=sameAuthors+1
        }
      }
    }
    return sameAuthors
  }

  def sameJournal(Journal1: String,Journal2: String):Int={
    if(Journal1==null || Journal2 ==null)
      return 0

    if(Journal1.compareTo(Journal2)==0){
      return 1
    }
    return 0
  }

  def YearDifference(Year1: Int, Year2: Int):Int={
    return math.abs(Year1 - Year2)
  }

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
    val dfHeaderless = sparkSession.read.format("csv").option("header", "false").load("Data-20181224/node_information.csv")
    val dfNodeInformation = dfHeaderless.toDF("Id", "Year", "Title", "Authors", "Journal", "Abstract")
    //load training set
    //dfNodeInformation.cache()
    val df2 = sparkSession.read.format("csv").option("header", "false").load("Data-20181224/training_set.txt").toDF("info")

    val dfComSplit = df2.withColumn("_tmp", split($"info", " ")).select(
      $"_tmp".getItem(0).as("Id1"),
      $"_tmp".getItem(1).as("Id2"),
      $"_tmp".getItem(2).as("label")
    ).drop("_tmp")//.limit(1000)

    val dfNodeInfoProstemp=TFIDF(dfNodeInformation,"Abstract","fAbstract",2000).drop("Abstract")


    val tokenizer = new Tokenizer().setInputCol("Title").setOutputCol("words")

    val word2Vec = new Word2Vec()
      .setInputCol("words")
      .setOutputCol("fTitle")
      .setVectorSize(20)
      .setMinCount(0)
    val w=tokenizer.transform(dfNodeInfoProstemp)
    val model2Vec = word2Vec.fit(w)

    val result = model2Vec.transform(w)

    //val Array(train, testing) = dfComSplit.limit(1000).randomSplit(Array[Double](0.8, 0.2), 18)
    //println(train.count(),testing.count())

    val users: RDD[(VertexId, (Int,String, String,Vector,Vector))] = result.rdd.map(x=>(x.getString(0).toLong,(x.getString(1).toInt,x.getString(3),x.getString(4),x.getAs[Vector](5),x.getAs[Vector](7))))
    val edgesTrain = dfComSplit.rdd.map(x => Edge(x.getString(0).toLong, x.getString(1).toLong,x.getString(2).toInt))
    val dftest = sparkSession.read.format("csv").option("header", "false").load("Data-20181224/training_set.txt").toDF("info")

    val dfComSplitTest = df2.withColumn("_tmp", split($"info", " ")).select(
      $"_tmp".getItem(0).as("Id1"),
      $"_tmp".getItem(1).as("Id2"),
      $"_tmp".getItem(2).as("label")
    ).drop("_tmp")//.limit(5000)

    val allEdges = dfComSplitTest.rdd.map(x=> Edge(x.getString(0).toLong, x.getString(1).toLong,2)) ++ edgesTrain
    //val edgesTest = testing.rdd.map(x=> Edge(x.getString(0).toLong, x.getString(1).toLong,2))
    //val edges = edgesTest++edgesTrain
    val u=VertexRDD(users)
    val graph = Graph(u,allEdges)


    val newAttributes: VertexRDD[(Int, String,String,Vector,Vector,Array[Long],Int)] = graph.aggregateMessages[(Int,String, String,Vector,Vector,Array[Long],Int)](
      triplet => { // Map Function
        if(triplet.attr==1) {
          triplet.sendToSrc((triplet.srcAttr._1, triplet.srcAttr._2, triplet.srcAttr._3, triplet.srcAttr._4,triplet.srcAttr._5, Array(triplet.dstId),0))
          triplet.sendToDst((triplet.dstAttr._1, triplet.dstAttr._2, triplet.dstAttr._3, triplet.dstAttr._4,triplet.dstAttr._5, Array(triplet.srcId),1))
        }else if(triplet.attr==0 || triplet.attr==2){
          triplet.sendToSrc((triplet.srcAttr._1, triplet.srcAttr._2, triplet.srcAttr._3, triplet.srcAttr._4,triplet.srcAttr._5, Array[Long](),0))
          triplet.sendToDst((triplet.dstAttr._1, triplet.dstAttr._2, triplet.dstAttr._3, triplet.dstAttr._4,triplet.dstAttr._5, Array[Long](),0))
        }
      },
      (a,b)=>(a._1,a._2,a._3,a._4,a._5,a._6++b._6,a._7+b._7)
    )
    val updatedGraph=Graph(newAttributes,allEdges)
    val facts=updatedGraph.triplets
    val readyRDD= facts.map(x=>(Row(x.srcId,x.srcAttr._1,x.srcAttr._2,x.srcAttr._3,x.srcAttr._4,x.srcAttr._5,x.srcAttr._6,x.srcAttr._7
      ,x.dstId,x.dstAttr._1,x.dstAttr._2,x.dstAttr._3,x.dstAttr._4,x.dstAttr._5,x.dstAttr._6,x.dstAttr._7,x.attr)))
    val aStruct = new StructType(Array(StructField("Id1",LongType,nullable = true)
      ,StructField("Year1",IntegerType,nullable = true)
      ,StructField("Authors1",StringType,nullable = true)
      ,StructField("Journal1",StringType,nullable = true)
      ,StructField("fAbstract1",VectorType,nullable = true)
      ,StructField("fTitle1",VectorType,nullable = true)
      ,StructField("Neigh1",ArrayType(LongType,true),nullable = true)
      ,StructField("InLinks1",IntegerType,nullable = true)
      ,StructField("Id2",LongType,nullable = true)
      ,StructField("Year2",IntegerType,nullable = true)
      ,StructField("Authors2",StringType,nullable = true)
      ,StructField("Journal2",StringType,nullable = true)
      ,StructField("fAbstract2",VectorType,nullable = true)
      ,StructField("fTitle2",VectorType,nullable = true)
      ,StructField("Neigh2",ArrayType(LongType,true),nullable = true)
      ,StructField("InLinks2",IntegerType,nullable = true)
      ,StructField("label",IntegerType,nullable = true)))
      //.toDF("Id1","Year1","Author1","Journal1","fAbstract1","fTitle1"
        //,"Id2","Year2","Author2","Journal2","fAbstract2","fTitle2")
    val readyDf = sparkSession.createDataFrame(readyRDD,aStruct)

    val jacdist= udf(jaccard _ )
    val sqdist = udf(sqrdist _ )
    val cosinedist = udf(cosdist _ )
    val clusterComparison = udf(comparison _ )
    val sAuthor= udf(sameAuthor _)
    val sJournal = udf(sameJournal _ )
    val difYear = udf(YearDifference _ )
    val sNeighbours = udf(Neighbours _ )
    val inLinksDif= udf(inLinksDifference _)
    //produce final dataframe calculate euclidean distance between abstracts and check if there is the same author between 2 articles
    val finalDf=readyDf.withColumn("Dist",cosinedist($"fAbstract1",$"fAbstract2"))
      .withColumn("DistTitle",cosinedist($"fTitle1",$"fTitle2"))
      .withColumn("YearDifference",difYear($"Year1",$"Year2"))
      //.withColumn("sameAuthors",sAuthor($"Authors1",$"Authors2"))
      .withColumn("Id",concat_ws(" ",$"Id1",$"Id2"))
      //.withColumn("sameJournal", sJournal($"Journal1",$"Journal2"))
      .withColumn("sameNeighbours",sNeighbours($"Neigh1",$"Neigh2"))
      .withColumn("LinksDifference",inLinksDif($"InLinks2",$"InLinks1"))
      .select("Id","label","YearDifference","Dist","DistTitle","sameNeighbours","LinksDifference")

    finalDf.persist(StorageLevel.MEMORY_AND_DISK)

    //finalDf.show()
    val trainingDf = finalDf.filter($"label"=!=2)
    val testingDf = finalDf.filter($"label"===2)
    //trainingDf.show()
    //testingDf.show()
    val assembler1 = new VectorAssembler().
      setInputCols(Array("Dist","sameNeighbours","YearDifference","DistTitle","LinksDifference")).
      setOutputCol("featuresTemp")

    //"Dist","DistTitle","sameAuthors","sameJournal","YearDifference","sameNeighbours"

    val assembledFinalDf = assembler1.transform(trainingDf).select("Id","featuresTemp","label")

    print("mid")
    import org.apache.spark.ml.feature.Normalizer
    val scaler = new Normalizer()
      .setInputCol("featuresTemp")
      .setOutputCol("features")
      .setP(1.0)

    val scaledData = scaler.transform(assembledFinalDf)

   /* MinMaxScaler
    val scaler = new MinMaxScaler()
      .setInputCol("featuresTemp")
      .setOutputCol("features")

    // Compute summary statistics and generate MinMaxScalerModel
    val scalerModel = scaler.fit(assembledFinalDf)

    // rescale each feature to range [min, max].
    val scaledData = scalerModel.transform(assembledFinalDf)*/

    //val Array(training, test) = scaledData.randomSplit(Array[Double](0.8, 0.2), 18)



    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setNumTrees(10)

    val trained=rf.fit(scaledData)

    val assembledFinalDfTest = assembler1.transform(testingDf).select("Id","featuresTemp","label")

    val scaledDataTest = scaler.transform(assembledFinalDfTest)

    val prediction=trained.transform(scaledDataTest)
    val ntesting=dfComSplitTest
      .withColumn("Idt",concat_ws(" ",$"Id1",$"Id2"))
      .withColumn("trueLabel",dfComSplitTest.col("label").cast(IntegerType))
      .select($"Idt",$"trueLabel")
    //ntesting.show()
    //prediction.show()
    val fPrediction= prediction.join(ntesting,$"Id"===$"Idt")
    //fPrediction.show()
    //decision tree model
    /*val dt = new DecisionTreeClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
    val trained=dt.fit(training)
    val prediction=trained.transform(test)*/
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("trueLabel")
      .setPredictionCol("prediction")
      .setMetricName("f1")
    val accuracy = evaluator.evaluate(fPrediction)
    println(s"F1 score = ${accuracy}")

  }
}

